
#include <stdio.h>
#include <sys/sysinfo.h>
#include <zlib.h>
#include <string>
#include <vector>
#include <array>
#include <pthread.h>
#include <fstream>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/types.h>

#include <synapse/Debug>
using namespace Syn;


// Original paper
// https://arxiv.org/pdf/2212.09410.pdf

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define LOG_INFO(...) { printf("[INFO] " __VA_ARGS__); }
#define LOG_ERROR(...) { printf("[ERROR] " __VA_ARGS__); }
#define ARRAY_LEN(x) (sizeof((x)) / sizeof((x)[0]))
#define CLASS_COUNT 4


// function prototypes
size_t zip(const char *_input_text, size_t len);

// data structures
typedef struct sample_t
{
    int class_ = -1;
    std::string clear_text;
    size_t len = 0;
    size_t compressed_size = 0;

    sample_t() {}
    sample_t(const std::string &_src, bool _compress=false)
    {
        clear_text = _src.substr(2);
        len = clear_text.length();
        class_ = atoi(&_src[0]) - 1;    // enforce zero-indexing of class
        if (_compress)
            compress();
    }

    void compress() { compressed_size = zip(clear_text.c_str(), len); }

} sample_t;

//
typedef struct ncd_t
{
    float distance;
    int class_;

} ncd_t;

//
typedef struct thread_info_t
{
    size_t sample_offset;
    size_t chunk_size;
    sample_t *input_sample;
    sample_t *samples_ptr;  // read and write, threaded
    ncd_t *ncds_ptr;
    
} thread_info_t;

//---------------------------------------------------------------------------------------
// function for sorting ncd_t using qsort
int cmp_ncd_distance(const void *_a, const void *_b)
{
    ncd_t *a = (ncd_t*)_a;
    ncd_t *b = (ncd_t*)_b;

    if (a->distance < b->distance) return -1;
    if (a->distance > b->distance) return  1;
    return 0;
}

//---------------------------------------------------------------------------------------
size_t zip(const char *_input_text, size_t _len)
{
    size_t n = _len + 1;
    size_t out_size = compressBound(n);
    uint8_t dest[out_size];
    compress2(dest, &out_size, (uint8_t *)_input_text, n, Z_BEST_COMPRESSION);
    return out_size;

}

//---------------------------------------------------------------------------------------
void parse_training_data(const char *_filename, std::vector<sample_t> &_training_samples)
{
    std::ifstream fin(_filename, std::ios::in | std::ios::app);
    std::string line;
    std::getline(fin, line);    // skip header
    size_t i = 0;
    while (std::getline(fin, line))
    {
        _training_samples.push_back(sample_t(line));
        i++;
    }

    fin.close();
    
}

//---------------------------------------------------------------------------------------
void parse_training_data2(const char *_filename, sample_t *_samples)
{
    std::ifstream fin(_filename, std::ios::in | std::ios::app);
    std::string line;
    std::getline(fin, line);    // skip header
    size_t i = 0;
    while (std::getline(fin, line))
    {
        _samples[i] = sample_t(line);
        i++;
    }

    fin.close();
    
}

//---------------------------------------------------------------------------------------
size_t line_count(const char *_filename)
{
    std::ifstream fin(_filename, std::ios::in | std::ios::app);
    std::string line;
    std::getline(fin, line);    // skip header
    size_t i = 0;
    while (std::getline(fin, line)) i++;
    
    fin.close();

    return i;
}

//---------------------------------------------------------------------------------------
void *compress_threaded(void *_thread_info)
{
    thread_info_t *ti = (thread_info_t*)_thread_info;
    size_t i0 = ti->sample_offset;
    size_t i1 = i0 + ti->chunk_size;
    
    // compress
    for (size_t i = i0; i < i1; i++)
        ti->samples_ptr[i].compress();

    return NULL;
        
}

//---------------------------------------------------------------------------------------
float ncd(sample_t *_x, sample_t *_y)
{
    // (from https://arxiv.org/pdf/2212.09410.pdf)
    //
    // NDC(x,y) = ( C(x|y) - min( C(x), C(y) ) ) / max( C(x), C(y) ),
    //
    // where C(x) is the compressed length of x (x|y is x and y concatenated).
    //
    
    // samples are of type sample_t, and thus already compressed at initialization
    // Cx = _x->compressed_size, Cy = _y->compressed_size
    //

    std::string xy = _x->clear_text + _y->clear_text;
    size_t Cxy = zip(xy.c_str(), xy.length());

    size_t min_xy = MIN(_x->compressed_size, _y->compressed_size);
    size_t max_xy = MAX(_x->compressed_size, _y->compressed_size);

    float ncd = (Cxy - min_xy) / (float)max_xy;
    
    return ncd;

}

//---------------------------------------------------------------------------------------
void *ncd_threaded(void *_thread_info)
{
    thread_info_t *ti = (thread_info_t*)_thread_info;
    size_t i0 = ti->sample_offset;
    size_t i1 = i0 + ti->chunk_size;
    
    // calculate ncd
    for (size_t i = i0; i < i1; i++)
    {
        ti->ncds_ptr[i].distance = ncd(ti->input_sample, &ti->samples_ptr[i]);
        ti->ncds_ptr[i].class_ = ti->samples_ptr[i].class_;
    }
    
    return NULL;
        
}

//---------------------------------------------------------------------------------------
void classify_sample(sample_t *_input_sample, sample_t *_samples, size_t _n, int _k)
{
    //
    int ncores = get_nprocs();
    pthread_t threads[ncores];
    thread_info_t tis[ncores];

    //
    ncd_t *ncds = (ncd_t *)malloc(sizeof(ncd_t) * _n);

    // for splitting work loads between the threads
    int chunk_size = _n / (size_t)ncores;
    int remain = _n - (ncores * chunk_size);
    
    LOG_INFO("(2) Compressing samples...\n");
    {
        Timer t("[INFO] threaded compression", true);
        for (int i = 0; i < ncores; i++)
        {
            int end = 0;
            if (i == ncores - 1) end = remain;
            
            tis[i].sample_offset    = i * chunk_size;
            tis[i].chunk_size       = chunk_size + end;
            tis[i].samples_ptr      = _samples;
            tis[i].input_sample     = _input_sample;
            tis[i].ncds_ptr         = ncds;

            pthread_create(&threads[i], NULL, compress_threaded, &tis[i]);

        }

        for (int i = 0; i < ncores; i++)
            pthread_join(threads[i], NULL);

    }

    LOG_INFO("(3) Calulating NCDs...\n");
    {
        Timer t("[INFO] threaded NCD calculation", true);
        for (int i = 0; i < ncores; i++)
            pthread_create(&threads[i], NULL, ncd_threaded, &tis[i]);
        
        for (int i = 0; i < ncores; i++)
            pthread_join(threads[i], NULL);

    }

    // 4. Classify using K-nearest neighbours
    //

    // sort ndcs by shortest distance and make the knn vote
    qsort(ncds, _n, sizeof(ncd_t), cmp_ncd_distance);

    int ks[CLASS_COUNT] = { 0 };
    for (size_t i = 0; i < _k; i++)
        ks[ncds[i].class_]++;

    // find max class
    int pred_class = -1;
    int max = -1;
    for (int i = 0; i < CLASS_COUNT; i++)
    {
        if (ks[i] > max)
        {
            max = ks[i];
            pred_class = i+1;
        }
    }

    LOG_INFO("(4) Classified sample as class %d.\n", pred_class);
    for (int i = 0; i < CLASS_COUNT; i++)
        LOG_INFO("    class %d: %d\n", i+1, ks[i]);

    free(ncds);

}

//---------------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    printf("===== Less is More: Parameter-Free Text Classification with Gzip =====\n");

    const char *train_file = "../data/train.csv";

    // allocate memory for training data
    size_t n = line_count(train_file);
    sample_t *samples = (sample_t *)malloc(sizeof(sample_t) * n);

    LOG_INFO("(1) Parsing training data...\n");
    parse_training_data2(train_file, samples);

    // input sample -- class 3
    const char *test_input = "Oil and Economy Cloud Stocks' Outlook (Reuters),Reuters - Soaring crude prices plus worries\about the economy and the outlook for earnings are expected to\hang over the stock market next week during the depth of the\summer doldrums.";
    sample_t input_sample(test_input, true);

    //
    classify_sample(&input_sample, samples, n, 500);

    free(samples);

    //
    return 0;
    
}

