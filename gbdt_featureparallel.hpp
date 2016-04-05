#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ctime>
#include <cassert>
#include <omp.h>
using namespace std;

// instance info
struct DFeature {
    vector<float> f; // features of this instance
    float y; // residuals for this instance
};

struct SFeature {
    vector< pair<int, float> > f;
    float y;
};

// tree node info
struct TNode
{    
    float value;
    float splitval; // split threshold
    int ind; //which feature be the split point
    int ch[2];
    float sum_y;
    float sum_sqr_y;
};

const float EPSI = 1e-4;

unsigned long long now_rand = 1;

double get_time() {    
    struct timeval   tp;
    struct timezone  tzp;
    gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );	
}

void set_rand_seed(unsigned long long seed)
{
    now_rand = seed;
}

unsigned long long get_rand()
{
    now_rand = ((now_rand * 6364136223846793005ULL + 1442695040888963407ULL) >> 1);
    return now_rand;
}

inline float sqr(const float &x) {
    return x * x;
}

inline int sign(const float &val) {
    if (val > EPSI) return 1;
    else if (val < -EPSI) return -1;
    else return 0;
}

double tot_parallel_time = 0.0;

class DecisionTree {
    private:
    // leaf node
    struct QNode {
        int nid;                
        //int left, right;
        int cnt;
        float err;
        
        QNode() {
            nid = cnt = 0;
        }
        
        /*
        QNode() {
            nid = left = right = 0;
        }        
        
        QNode(const int &nid_, const int &left_, const int &right_) {
            nid = nid_;
            left = left_;            
            right = right_;
        }
        */
        QNode(const int &nid_, const int &cnt_) {
            nid = nid_;
            cnt = cnt_;
            err = 0.0f;
        }
        
        QNode(const int &nid_, const int &cnt_, const float &err_) {
            nid = nid_;
            cnt = cnt_;
            err = err_;
        }
    };    
    
    struct SplitInfo {
        int bind; // feature id
        float bsplit; // split threshold
        int cnt[2]; // num of instances for right and left
        float sum_y[2], sum_sqr_y[2];
        float err;
        
        void update(const SplitInfo &sinfo) {
            bind = sinfo.bind;
            bsplit = sinfo.bsplit;
            cnt[0] = sinfo.cnt[0]; cnt[1] = sinfo.cnt[1];
            sum_y[0] = sinfo.sum_y[0]; sum_y[1] = sinfo.sum_y[1];
            sum_sqr_y[0] = sinfo.sum_sqr_y[0]; sum_sqr_y[1] = sinfo.sum_sqr_y[1];
            err = sinfo.err;
        }
    };
    
    struct ThreadInfo {
        int cnt0; // right num of instances for split
        float sum0, ss0;
        float last_val;
        SplitInfo spinfo;
    };
    
    vector<QNode> q; // all leaf nodes for one level

    vector<SplitInfo> split_infos; // split info for each QNode in q
    
    float *y_list, *sqr_y_list;
    int *positions;
    
    vector<DFeature> *features_ptr;
    
    public:
    vector<TNode> tree;
    
    int min_children;
    int max_depth;
    
    int n; // number of instances
    int m; // number of features
    int nthread; // number of threads    
    
    
    
    private:
    void init_data() {
        q.reserve(256);
        q.resize(0);
        split_infos.reserve(256);
        split_infos.resize(0);
        tree.reserve(256);
        tree.resize(0);
        
        omp_set_num_threads(nthread);
        
        #pragma omp parallel
        {
            this->nthread = omp_get_num_threads();
        }
        printf("number of thread: %d\n", this->nthread);
    }       
    
    void update_queue() {
        vector<DFeature> &features = *features_ptr;
        vector<QNode> new_q; // all leaf node
        TNode new_node;
        vector< pair<int, int> > children_q_pos(q.size());
        for (int i = 0; i < q.size(); i++) {
            /*
            //printf("nid: %d, left: %d, right: %d\n", q[i].nid, q[i].left, q[i].right);
            printf("nid: %d, cnt %d\n", q[i].nid, q[i].cnt);
            printf("bind: %d, bsplit: %f, cnt0: %d, cnt1: %d\n", split_infos[i].bind, 
                split_infos[i].bsplit, split_infos[i].cnt[0], split_infos[i].cnt[1]);
            printf("sum0: %f, sum1: %f, v0: %f, v1: %f\n", split_infos[i].sum_y[0], 
                split_infos[i].sum_y[1], split_infos[i].sum_y[0] / max(1, split_infos[i].cnt[0]),
                split_infos[i].sum_y[1] / max(1, split_infos[i].cnt[1]));
            printf("\n");
            */
            if (split_infos[i].bind >= 0) {
                // put the final split info to tree
                int ii = q[i].nid; // ith of QNode is iith of tree
                tree[ii].ind = split_infos[i].bind;                        
                tree[ii].splitval = split_infos[i].bsplit;                
                tree[ii].ch[0] = tree.size();
                tree[ii].ch[1] = tree.size() + 1;
                children_q_pos[i].first = new_q.size();
                children_q_pos[i].second = new_q.size() + 1;
                
                //new_q.push_back(QNode(tree.size(), split_infos[i].cnt[0], split_infos));
                //new_q.push_back(QNode(tree.size() + 1, split_infos[i].cnt[1]));
                                                
                // create two leaf node of tree and two QNode with info
                for (int c = 0; c < 2; c++) {
                    new_node.ind = -1;
                    new_node.value = split_infos[i].sum_y[c] / split_infos[i].cnt[c]; // mean of sum_y
                    new_node.sum_y = split_infos[i].sum_y[c];
                    new_node.sum_sqr_y = split_infos[i].sum_sqr_y[c];
                    float err = new_node.sum_sqr_y - new_node.sum_y*new_node.sum_y/split_infos[i].cnt[c];
                    new_q.push_back(QNode(tree.size(), split_infos[i].cnt[c], err));
                    tree.push_back(new_node);
                }
            }
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            int &pos = positions[i];
            if (pos >= 0 && split_infos[pos].bind >= 0) {
                if (features[i].f[split_infos[pos].bind] <=  split_infos[pos].bsplit) {
                    pos = children_q_pos[pos].first;
                } else {
                    pos = children_q_pos[pos].second;
                }                
            } else pos = -1;
        }
        
        q = new_q;
    }
       
    // set initial value and sort the column feature list
    void initial_column_feature_list(vector< vector< pair<float, int> > > &col_fea_list, vector<int> &id_list) {
        
        vector<DFeature> &features = *features_ptr; 
        
        col_fea_list.resize(m); // m features
                
        for (int i = 0; i < m; i++) {
            col_fea_list[i].resize(id_list.size()); // each feature has id_list.size() instances
        }
        
        #pragma omp parallel for schedule(static)    
        for (int i = 0; i < id_list.size(); i++) {
            int ins_id = id_list[i];
            for (int j = 0; j < m; j++) {
                col_fea_list[j][i].first = features[ins_id].f[j];
                col_fea_list[j][i].second = ins_id; 
            }
        }
        
        #pragma omp parallel for schedule(dynamic,1)
        for (int i = 0; i < m; i++) {
            sort(col_fea_list[i].begin(), col_fea_list[i].end()); // sort value of all instance for each feature
        }        
    }       
    
    // linear search for best split
    void find_split(int fid, vector< pair<float, int> > &fea_list, vector<ThreadInfo> &tinfo_list) {
        
        // each q node
        for (int i = 0; i < tinfo_list.size(); i++) {
            tinfo_list[i].cnt0 = 0;        
            tinfo_list[i].sum0 = 0.0f;
            tinfo_list[i].ss0 = 0.0f;
        }
        
        float ss1, sum1, err;
        int top = 0;
        // each instances
        for (int i = 0; i < fea_list.size(); i++) {
            int iid = fea_list[i].second; // instance id
            int pos = positions[iid]; // which node of q belong to
            if (pos < 0) continue;
            fea_list[top++] = fea_list[i];
            int nid = q[pos].nid; // node id of whole tree
            ThreadInfo &tinfo = tinfo_list[pos];
            // |--10--|-----split available-----|--10--|
            if (tinfo.cnt0 >= min_children && q[pos].cnt - tinfo.cnt0 >= min_children && sign(fea_list[i].first - tinfo.last_val) != 0) {
                float &sum0 = tinfo.sum0;
                float &ss0 = tinfo.ss0;
                sum1 = tree[nid].sum_y - sum0;
                ss1 = tree[nid].sum_sqr_y - ss0;
                err = ss0 - sum0 * sum0 / tinfo.cnt0 + ss1 - sum1 * sum1 / (q[pos].cnt-tinfo.cnt0);
                // save new split info for this node
                if (sign(err - tinfo.spinfo.err) < 0) {
                    SplitInfo &tbest = tinfo.spinfo;
                    tbest.err = err;
                    tbest.bind = fid;
                    tbest.bsplit = (fea_list[i].first + tinfo.last_val) / 2; // split threshold
                    tbest.sum_y[0] = sum0; tbest.sum_y[1] = sum1;
                    tbest.sum_sqr_y[0] = ss0; tbest.sum_sqr_y[1] = ss1;
                    tbest.cnt[0] = tinfo.cnt0; tbest.cnt[1] = q[pos].cnt - tinfo.cnt0;
                }
            }
            tinfo.cnt0 += 1;
            tinfo.sum0 += y_list[iid];
            tinfo.ss0 += sqr_y_list[iid];                                               
            tinfo.last_val = fea_list[i].first;            
        }
        fea_list.resize(top);
    }
        
    public:
    DecisionTree(vector<DFeature> &features, int max_depth, int max_feature, int max_pos, 
        int min_children, float bootstrap, int nthread) {
        
        this->n = features.size();        
        this->m = features.size() > 0 ? features[0].f.size() : 0;
        this->min_children = max(min_children, 1);
        this->max_depth = max_depth;
        this->nthread = nthread ? nthread : 1;
        this->features_ptr = &features;
        
        init_data();
        
        vector<int> id_list; // instance num. of train data
        id_list.reserve(n);
        float sum_y = 0.0;
        float sum_sqr_y = 0.0;
        int tcnt = 0;
        
        y_list = new float[n];
        sqr_y_list = new float[n];
        positions = new int[n]; // whether this instance is used
            
        // process bootstrap
        for (int i = 0; i < n; i++) {
            if ((float)get_rand() / RAND_MAX >= bootstrap) {
                id_list.push_back(i); // choice ith instance as train data
                y_list[i] = features[i].y;
                sqr_y_list[i] = sqr(features[i].y);
                sum_y += y_list[i];
                sum_sqr_y += sqr_y_list[i];                
                positions[i] = 0;
            } else {
                positions[i] = -1;
            }
        }        
        
        // add the root node        
        TNode node;
        node.ind = -1;
        node.value = sum_y / (id_list.size() ? id_list.size() : 1);
        node.sum_y = sum_y;
        node.sum_sqr_y = sum_sqr_y;        
        tree.push_back(node);
        
        if (id_list.size() == 0) return;
        q.push_back(QNode(0, id_list.size(), sum_sqr_y-sum_y*sum_y/id_list.size()));  
        
        // set initial value and sort the column feature list
        vector< vector< pair<float, int> > > col_fea_list;
        initial_column_feature_list(col_fea_list, id_list);
        printf("initial column feature done...\n");        
        
        vector< vector<ThreadInfo> > tinfos(nthread);
        
        // build a decision tree         
        for (int dep = 0; dep < max_depth; dep++) {
            if (q.size() == 0) break;
            //printf("building depth %d...\n", dep);
            
            int nq = q.size();
            split_infos.resize(q.size());
            //printf("size of split info: %d\n", split_infos.size());
            
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nthread; i++) {
                tinfos[i].resize(q.size());                
                for (int j = 0; j < q.size(); j++) {                    
                    tinfos[i][j].spinfo.bind = -1;
                    //tinfos[i][j].spinfo.err = 1e30;
                    tinfos[i][j].spinfo.err = q[j].err;
                }
            }
            
            //printf("initialize thread info done...\n");
            
            #pragma omp parallel for schedule(dynamic,1)
            for (int fid = 0; fid < m; fid++) {         
                const int tid = omp_get_thread_num();
                find_split(fid, col_fea_list[fid], tinfos[tid]);
            }
            
            //printf("find split done...\n");
            
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nq; i++) {
                SplitInfo &spinfo = split_infos[i];
                spinfo.bind = -1;
                for (int j = 0; j < nthread; j++)
                    if (tinfos[j][i].spinfo.bind >= 0 && (spinfo.bind < 0 || spinfo.err > tinfos[j][i].spinfo.err))
                        spinfo.update(tinfos[j][i].spinfo);
            }
            
            //printf("merge split info done.\n");
            
           /* update tree nodes
           (assign split info and create new leaf nodes) 
           and q nodes(new leaf nodes) */
            update_queue();
            
            //printf("update queue done...\n");
        }    
                   
        delete[] y_list;
        delete[] sqr_y_list;
        delete[] positions;
#ifdef cpp11        
        tree.shrink_to_fit();
#endif     
    }
        
    float predictTree(vector<float> &f) {
        int n = 0;
        while (tree[n].ind >= 0) // whether tree[n] is leaf node
        {
            if (f[ tree[n].ind ] <= tree[n].splitval)
                n = tree[n].ch[0];
            else
                n = tree[n].ch[1];
        }
        return tree[n].value;
    }           
};

// cal rmse for evaluation
float cal_rmse(vector<float> &pred, vector<float> &gt) {
    assert(pred.size() == gt.size());
    float rmse = 0;
    for (int i = 0; i < pred.size(); i++) {
        rmse += sqr(pred[i] - gt[i]);
    }
    rmse = sqrt(rmse / pred.size());
    return rmse;
}

// cal auc for evaluation
float cal_auc(vector<float> &pred, vector<float> &gt) {
    assert(pred.size() == gt.size());
    vector< pair<float, float> > tv;
    for (int i = 0; i < pred.size(); i++)
        tv.push_back(make_pair(pred[i], -gt[i]));
    sort(tv.begin(), tv.end());
    for (int i = 0; i < tv.size(); i++) 
        tv[i].second = -tv[i].second;
    int pos_cnt = 0, neg_cnt = 0;
    float cor_pair = 0;
    for (int i = 0; i < tv.size(); i++)
        if (tv[i].second > 0.5) {
            pos_cnt++;
            cor_pair += neg_cnt;
        } else {
            neg_cnt++;
        }
    return (neg_cnt > 0 && pos_cnt > 0) ? (cor_pair / pos_cnt / neg_cnt) : 0.0;
}

// create the boost forest
class BoostedForest {
    public:
    vector<DecisionTree*> trees;
    int depth, max_feature, max_pos, min_children, nthread, num_tree;
    float bootstrap, step;
    vector<float> cur_vals, ori_vals;
    vector<float> steps;
    vector<DFeature> *val_features_ptr;

    BoostedForest() {
        val_features_ptr = NULL;
        step = 0.1;
        depth = 5;
        max_feature = max_pos = -1;
        min_children = 10;
        nthread = 1;
        bootstrap = 0;
        num_tree = 50;
    }
    
    void set_val_data(vector<DFeature> &data) {
        val_features_ptr = &data;
    }

    void set_params(const char *name, const char *val) {
        if (!strcmp(name, "num_tree")) num_tree = atoi(val);
        if (!strcmp(name, "depth")) depth = atoi(val);
        if (!strcmp(name, "min_children")) min_children = atoi(val);
        if (!strcmp(name, "nthread")) nthread = atoi(val);
        if (!strcmp(name, "step")) step = static_cast<float> (atof(val));
        if (!strcmp(name, "bootstrap")) bootstrap = static_cast<float> (atof(val));
    }   
 
    void buildForest(vector<DFeature> &features) {        
                
        if (max_feature < 0) max_feature = int(sqrt(features[0].f.size()) + 1);

        // load train data
        cur_vals = vector<float>(features.size());
        ori_vals = vector<float>(features.size());
        for (int i = 0; i < features.size(); i++)
            ori_vals[i] = features[i].y;
        
        // load validation data
        vector<float> val_vals;
        vector<float> pred_vals;
        if (val_features_ptr != NULL) {
            vector<DFeature> &val_features = *val_features_ptr;
            pred_vals = vector<float>(val_features.size());
            val_vals = vector<float>(val_features.size());
            for (int i = 0; i < val_features.size(); i++)
                val_vals[i] = val_features[i].y;
        }        
        
        float train_rmse = -1, test_rmse = -1;
        float train_auc = -1, test_auc = -1;        
                
        double train_time = 0.0;
        double start_time = get_time();
        // for each tree
        for (int i = 0; i < num_tree; i++)
        {
            double iter_start_time = get_time();
            // update residual for each instance
            for (int j = 0; j < features.size(); j++)
                features[j].y = ori_vals[j] - cur_vals[j];
            // create a decision tree, the only different is features.y(residual)
            DecisionTree *dt = new DecisionTree(features, depth, max_feature, max_pos, min_children, bootstrap, nthread);
            trees.push_back(dt);
            
            for (int j = 0; j < features.size(); j++) {
                cur_vals[j] += dt->predictTree(features[j].f) * step; // pred of this instance sum(leaf node value of all trees)
            }    
            
            train_time += get_time() - iter_start_time;
            
           // evaluation
            train_rmse = cal_rmse(cur_vals, ori_vals);
            train_auc = cal_auc(cur_vals, ori_vals);
            
            if (val_features_ptr != NULL) {                
                vector<DFeature> &val_features = *val_features_ptr;
                for (int j = 0; j < val_features.size(); j++) {
                    pred_vals[j] += dt->predictTree(val_features[j].f) * step;                    
                }
                test_rmse = cal_rmse(pred_vals, val_vals);
                test_auc = cal_auc(pred_vals, val_vals);
            }
            
            steps.push_back(step);
            
            printf("iter: %d, train_rmse: %.6lf, test_rmse: %.6lf, tree_size: %d\n", i + 1, train_rmse, test_rmse, dt->tree.size());
            printf("train_auc: %.6lf, test_auc: %.6lf\n", train_auc, test_auc);            
            printf("%.3f seconds passed, %.3f seconds in parallel,%.3f seconds in training\n", get_time() - start_time, tot_parallel_time, train_time);
        }
                
        FILE *fout = fopen("time.out", "w");
        fprintf(fout, "%.3f\n", train_time);
        fclose(fout);
        
        for (int j = 0; j < features.size(); j++)
            features[j].y = ori_vals[j];
    }

    void addTree(vector<DFeature> &features) {        
        addTree(features, 1);
    }

    void addTree(vector<DFeature> &features, int treecnt) {        
        for (int j = 0; j < features.size(); j++) {
            ori_vals[j] = features[j].y;            
        }
        while (treecnt--) {
            for (int j = 0; j < features.size(); j++) {                
                features[j].y = ori_vals[j] - cur_vals[j];
            }
            DecisionTree *dt = new DecisionTree(features, depth, max_feature, max_pos, min_children, bootstrap, nthread);
            trees.push_back(dt);                          
            for (int j = 0; j < features.size(); j++) {
                cur_vals[j] += dt->predictTree(features[j].f) * step;
            }
            steps.push_back(step);
        }
        for (int j = 0; j < features.size(); j++) {            
            features[j].y = ori_vals[j];
        }
    }
    
    void set_step(float step_) {
        step = step_;
    }
    
    float predictForest(vector<float> &f) {
        float ret = 0;        
        for (int j = 0; j < trees.size(); j++) {
            ret += trees[j]->predictTree(f) * steps[j];            
        }        
        return ret;
    }
};
