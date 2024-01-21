
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

#include "utils.h"

#include <synapse/Debug>
using namespace Syn;


// Original paper
// https://arxiv.org/pdf/2212.09410.pdf

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define ARRAY_LEN(a) (sizeof((a)) / sizeof((a[0])))

#define CLASS_COUNT              4


// function prototypes
size_t zip(uint8_t *_input_text, size_t len);

// data structures
typedef struct sample_t
{
    int class_ = -1;
    uint8_t *clear_text = nullptr;
    int len = 0;
    size_t compressed_size = 0;

    sample_t() {}
    sample_t(const char *_src)              { init(_src, strlen(_src)); }
    sample_t(const char *_src, size_t _n)   { init(_src, _n); }

    void init(const char *_src, size_t _n)
    {
        clear_text = new uint8_t[_n];
        len = _n;
        memcpy(clear_text, _src+2, _n); // skip leading class and '.'
        class_ = atoi(&_src[0]) - 1;    // enforce zero-indexing of class
        compressed_size = zip(clear_text, _n);
    }

    void compress() { compressed_size = zip(clear_text, len); }

    //
    // ~sample_t() { delete clear_text; }   // not needed, stored in std::vector<>

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
    std::vector<sample_t> *samples_v_ptr;
    // std::vector<ncd_t> *ncds_v_ptr;
    ncd_t *ncds_ptr;

} thread_info_t;

// function for sorting ncd_t using std::sort
int cmp_ncd_distance(const void *_a, const void *_b)
{
    ncd_t *a = (ncd_t*)_a;
    ncd_t *b = (ncd_t*)_b;

    if (a->distance < b->distance) return -1;
    if (a->distance > b->distance) return  1;
    return 0;
}
bool cmp_ncd(ncd_t &_a, ncd_t &_b) { return (_a.distance < _b.distance); }

//---------------------------------------------------------------------------------------
size_t zip(uint8_t *_input_text, size_t len)
{
    size_t out_size = compressBound(len);
    uint8_t dest[out_size];
    // size_t out_size;
    // uint8_t dest[8192];
    compress2(dest, &out_size, (uint8_t *)_input_text, len, Z_BEST_COMPRESSION);
    return out_size;

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

    // uint8_t *xy = new uint8_t[_x->len + _y->len];
    uint8_t xy[2048];
    memcpy(xy, _x->clear_text, _x->len);
    memcpy(xy + _x->len, _y->clear_text, _y->len);
    size_t Cxy = zip(xy, _x->len + _y->len);

    size_t min_xy = MIN(_x->compressed_size, _y->compressed_size);
    size_t max_xy = MAX(_x->compressed_size, _y->compressed_size);

    float ncd = (Cxy - min_xy) / (float)max_xy;
    
    return ncd;

}

//---------------------------------------------------------------------------------------
void *classify_sample_threaded(void *_thread_info)
{
    thread_info_t *ti = (thread_info_t*)_thread_info;

    size_t start = ti->sample_offset;
    size_t stop = start + ti->chunk_size;

    // pid_t tid = syscall(__NR_gettid);
    // LOG_INFO("thread %d: start = %zu, stop = %zu\n", tid, start, stop);

    // 1. calculate the compression length of all training samples
    //for (size_t i = start; i < stop; i++)
    //    (*ti->samples_v_ptr)[i].compress();
    
    // 2. calculate NCD between input sample and allotted training samples
    for (size_t i = start, j = 0; i < stop; i++, j++)
    {
        // (*ti->ncds_v_ptr)[i].distance = ncd(ti->input_sample, &(*ti->samples_v_ptr)[i]);
        // (*ti->ncds_v_ptr)[i].class_ = (*ti->samples_v_ptr)[i].class_;
        ti->ncds_ptr[j].distance = ncd(ti->input_sample, &(*ti->samples_v_ptr)[i]);
        ti->ncds_ptr[j].class_ = (*ti->samples_v_ptr)[i].class_;
    }

    return NULL;

}

//---------------------------------------------------------------------------------------
void classify_sample(sample_t _input_sample, 
                     std::vector<sample_t> &_training_samples, 
                     size_t _k)
{
    Timer t;

    // thread it!
    int ncores = get_nprocs();
    LOG_INFO("%d cores detected.\n", ncores);
    
    pthread_t threads[ncores];
    thread_info_t tis[ncores];

    size_t n = _training_samples.size();
    // std::vector<ncd_t> ncds(_training_samples.size());
    ncd_t *ncds = new ncd_t[n];

    // for splitting work loads between the threads
    int chunk_size = n / (size_t)ncores;
    int remain = n - (ncores * chunk_size);

    for (int i = 0; i < ncores; i++)
    {
        int end = 0;
        if (i == ncores - 1) end = remain;
        
        tis[i].sample_offset    = i * chunk_size;
        tis[i].chunk_size       = chunk_size + end;
        tis[i].samples_v_ptr    = &_training_samples;
        tis[i].ncds_ptr         = new ncd_t[chunk_size];    // delete[]d at pthread_join
        tis[i].input_sample     = &_input_sample;

        pthread_create(&threads[i], NULL, classify_sample_threaded, &tis[i]);

    }

    // collect and concatenate NCDs
    for (int i = 0; i < ncores; i++)
    {
        pthread_join(threads[i], NULL);
        // append ncds from thread
        memcpy(ncds + i * chunk_size, tis[i].ncds_ptr, sizeof(ncd_t) * chunk_size);
        delete[] tis[i].ncds_ptr;
    }

    // sort ndcs by shortest distance and make the knn vote
    // std::sort(ncds.begin(), ncds.end(), cmp_ncd);
    qsort(ncds, n, sizeof(ncd_t), cmp_ncd_distance);
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

    LOG_INFO("Classified sample as class %d (%f ms).\n", pred_class, t.getDeltaTimeMs());
    for (int i = 0; i < CLASS_COUNT; i++)
        LOG_INFO("    class %d: %d\n", i+1, ks[i]);

    delete[] ncds;

}

//---------------------------------------------------------------------------------------
void parse_training_data(const char *_filename, std::vector<sample_t> &_training_samples)
{
    std::ifstream fin(_filename, std::ios::in | std::ios::app);
    std::string line;
    std::getline(fin, line);    // skip header
    int i = 0;
    while (std::getline(fin, line))
        // _training_samples[i++] = sample_t(line.c_str(), line.size());    // slow
        _training_samples.push_back(sample_t(line.c_str(), line.size()));   // fast

    fin.close();
    
}

//---------------------------------------------------------------------------------------
size_t line_count(const char *_filename)
{
    std::ifstream fin(_filename, std::ios::in | std::ios::app);
    std::string line;
    std::getline(fin, line);    // skip header
    size_t i = 0;
    while (std::getline(fin, line))
        i++;
    fin.close();

    return i;
}

//---------------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    printf("===== Less is More: Parameter-Free Text Classification with Gzip =====\n");

    const char *train_file = "../data/train.csv";

    // allocate memory for training data
    size_t n = line_count(train_file);
    LOG_INFO("%zu samples in '%s'.\n", n, train_file);
    std::vector<sample_t> train_samples(n);
    
    LOG_INFO("Parsing and compressing training data (1)...\n");
    {
        Timer t;
        parse_training_data(train_file, train_samples);
        LOG_INFO("Processed training data in %f ms.\n", t.getDeltaTimeMs());
    }
    // Don't know why, but it's faster to parse (and compress) the training data if 
    // using .push_back and then moving the data back and resizing compared to index-
    // accessing the vector (see in parse_training_data()).
    //
    memmove(train_samples.data(), train_samples.data()+n, sizeof(sample_t)*n);
    train_samples.resize(n);

    // input sample -- class 3
    const char *test_input = "Oil and Economy Cloud Stocks' Outlook (Reuters),Reuters - Soaring crude prices plus worries\about the economy and the outlook for earnings are expected to\hang over the stock market next week during the depth of the\summer doldrums.";
    sample_t input_sample(test_input);

    // predict class
    classify_sample(input_sample, train_samples, 200);

    //
    return 0;
    
}

