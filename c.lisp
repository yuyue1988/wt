
(defun group (lst n)
	   (if (zerop n)
	       "zero length is not permitted"
	       (labels ((rec (x acc) 
			     (if (nthcdr n x)
				 (append (rec (nthcdr n x) acc) (list (subseq x 0 n)))
				 (append acc (list x)))))
		 (reverse (rec lst nil)))))

(defun flattern (lst)
	   (labels ((rec (x acc)
		      (cond ((null x) acc)
			    ((atom x) (cons x acc))
			    (t (rec (car x) 
				    (rec (cdr x) acc))))))
	     (rec lst nil)))

(defun mkstr (&rest args)
	   (with-output-to-string (s)
	     (dolist (a args) (princ a s))))



FM::FM(int class_n, int fm_dim, int fm_n, valueType rand_std,
       bool donotInit) :
    parameters(vector<fmPara<valueType>>(
        class_n - 1,
        fmPara<valueType>(fm_dim, fm_n, 0.0f))),
    class_num(class_n), dimension(fm_dim), factor_n(fm_n),
    usage(vector<fmPara<bool>>(class_n - 1, fmPara<bool>(fm_dim, fm_n, false))),
    sigma2(vector<valueType>(fm_dim, 1.0)) {

  if (!donotInit) {
    opt_buf =
        vector<fmPara<array<valueType, optBufSize>>>(
            class_num - 1,
            fmPara<array<valueType, optBufSize>>
            (dimension,
             factor_n,
            {0.0f, 0.0f, 0.0f}));

    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<valueType> distr(0.0f, rand_std);
    for (int i = 0; i < class_num - 1; i++) {
      for (int j = 0; j < dimension; j++) {
        for (int k = 0; k < factor_n; k++) {
          parameters[i].v[j][k] = distr(generator);
        }
      }
    }
  }
  records = 0;
}

valueType FM::fmFunc(const X_type& x, int class_index,
                     vector<valueType>& sums,
                     vector<valueType>& squareSums) {
  valueType result = parameters[class_index].w0;
  for (size_t i = 0; i < x.size(); i++) {
    int fea_index = x[i].first;
    valueType fea_value = x[i].second;
    result += parameters[class_index].w[fea_index] * fea_value;
    for (int j = 0; j < factor_n; j++) {
      valueType tm = parameters[class_index].v[fea_index][j] * fea_value;
      if (i == 0) {
        sums[j] = tm;
        squareSums[j] = tm * tm;
      } else {
        sums[j] += tm;
        squareSums[j] += tm * tm;
      }
    }
  }

  for (int j = 0; j < factor_n; j++) {
    result += (sums[j] * sums[j] - squareSums[j]) * 0.5f;
  }
  
  return result;
}

vector<valueType> FM::predict(const X_type& x) {
  vector<valueType> result(class_num, 0.0);
  vector<valueType> exp_parts(class_num, 1.0);
  vector<valueType> sum_cache(factor_n);
  vector<valueType> square_sum_cache(factor_n);
  valueType sum = 1.0;
  for (int i = 0; i < class_num - 1; i++) {
    valueType mValue = fmFunc(x, i, sum_cache, square_sum_cache);
    exp_parts[i] = safeExp(-1.0 * mValue);
    sum += exp_parts[i];
  }
  for (int i = 0; i < class_num; i++) {
    result[i] = exp_parts[i] / sum;
  }
  return result;
}

vector<valueType> FM::giveMVals(const X_type& x) {
  vector<valueType> sum_cache(factor_n);
  vector<valueType> square_sum_cache(factor_n);
  vector<valueType> result(class_num - 1);
  for (int i = 0; i < class_num - 1; i++) {
    result[i] = fmFunc(x, i, sum_cache, square_sum_cache);
  }
  return result;
}

valueType FM::loss(const X_type& x, int class_index) {
  vector<valueType> pred = predict(x);
  return -safeLog(pred[class_index]);
}

void FM::paraUpdate(const X_type& x, int class_index,
                    valueType sample_weight, Optimizer* optimizer,
                    LossObj* lossobj,
                    vector<vector<valueType>>& sum_cache,
                    vector<valueType>& exp_parts,
                    vector<valueType>& squareSums,
                    vector<fmPara<valueType>>* grad_rec) {
  exp_parts[class_num - 1] = 1.0f;
  valueType sum = 1.0;
  for (int i = 0; i < class_num - 1; i++) {
    valueType mValue = fmFunc(x, i, sum_cache[i], squareSums);
    exp_parts[i] = safeExp(-1.0 * mValue);
    sum += exp_parts[i];
  }
  for (int i = 0; i < class_num - 1; i++) {
    valueType pref = (class_index == i ? 1.0f : 0.0f);
    valueType dl_dw0 =
        lossobj->dl_dT(exp_parts, sum, class_index, i) * sample_weight;
    if (grad_rec != nullptr) {
      (*grad_rec)[i].w0 = dl_dw0;
    }
    usage[i].w0 =
        optimizer->optProcess(&opt_buf[i].w0,
                               &parameters[i].w0, dl_dw0);

    for (size_t j = 0; j < x.size(); j++) {
      int fea_index = x[j].first;
      valueType fea_value = x[j].second;
      valueType dl_dw = dl_dw0 * fea_value;
      if (grad_rec != nullptr) {
        (*grad_rec)[i].w[fea_index] = dl_dw;
      }
      usage[i].w[fea_index] =
          optimizer->optProcess(&opt_buf[i].w[fea_index],
                                 &parameters[i].w[fea_index],
                                 dl_dw);
      for (int k = 0; k < factor_n; k++) {
        valueType v = parameters[i].v[fea_index][k];
        valueType dl_dv = dl_dw0 * (sum_cache[i][k] - v * fea_value) * fea_value;
        if (grad_rec != nullptr) {
          (*grad_rec)[i].v[fea_index][k] = dl_dv;
        }
        usage[i].v[fea_index][k] =
            optimizer->optProcess(&opt_buf[i].v[fea_index][k],
                                   &parameters[i].v[fea_index][k],
                                   dl_dv);
      }
    }
  }
  // update sigma^2
  for (size_t j = 0; j < x.size(); j++) {
    int fea_index = x[j].first;
    int fea_value = x[j].second;
    sigma2[fea_index] += fea_value * fea_value;
  }
  records++;
}

template<class T>
struct fmPara {
  T w0;
  vector<T> w;
  vector<vector<T>> v;
  fmPara(int dim, int n, T init) :
    w0(init),
    w(vector<T>(dim, init)),
    v(vector<vector<T>>(dim, vector<T>(n, init))) {
  }
};

class FM {
 public:
  FM(int class_n, int fm_dim, int fm_n, valueType rand_std = 1.0f / 200,
     bool donotInit = false);
  vector<valueType> predict(const X_type& x);
  vector<valueType> giveMVals(const X_type& x);
  void paraUpdate(const X_type& x, int class_index,
                  valueType sample_weight, Optimizer* optimizer,
                  LossObj* lossobj,
                  vector<vector<valueType>>& sum_cache,
                  vector<valueType>& exp_parts,
                  vector<valueType>& squareSums,
                  vector<fmPara<valueType>> * grad_rec = nullptr);
  valueType loss(const X_type& x, int class_index);
  void saveModel(const string& fn, bool store_confidence = false);
  static FM loadModel(const string& fn);

  vector<fmPara<valueType>> parameters;
  vector<fmPara<bool>> usage;
  int class_num;
  int factor_n;
  int dimension;

 private:
  vector<fmPara<array<valueType, optBufSize>>> opt_buf;
  vector<valueType> sigma2;
  long long int records;
  valueType fmFunc(const X_type& x, int class_index,
		   vector<valueType>& sums,
		   vector<valueType>& squareSums);
};
  
  
  
  
  class LossObj {
 public:
  virtual valueType dl_dT(const vector<valueType>& exp_parts,
                          const valueType sum,
                          int label_index,
                          int target_index) = 0;
  virtual ~LossObj() {}
};

class negtiveLogLoss : public LossObj {
 public:
  negtiveLogLoss();
  valueType dl_dT(const vector<valueType>& exp_parts,
                  const valueType sum,
                  int label_index,
                  int target_index);
};

class l2Loss : public LossObj {
 public:
  l2Loss();
  valueType dl_dT(const vector<valueType>& exp_parts,
                  const valueType sum,
                  int label_index,
                  int target_index);
};


l2Loss::l2Loss() {}

valueType l2Loss::dl_dT(const vector<valueType>& exp_parts,
                        const valueType sum,
                        int label_index,
                        int target_index) {
  valueType pred = exp_parts[label_index] / sum;
  valueType tmp = 1.0f - pred;
  valueType ret = 2.0 * pred * tmp;
  ret *= (label_index == target_index ? tmp : -exp_parts[target_index] / sum);
  return ret;
}
  
  
  
  bool ftrlOptimizer::optProcess(array<valueType, optBufSize>* buf,
                               valueType* W, const valueType& g) {
  valueType& q = (*buf)[0];
  valueType& z = (*buf)[1];
  
  valueType sigma = (sqrt(q + g * g) - sqrt(q)) / alpha;
  q = q + g * g;
  z = z + g - (*W) * sigma;

  bool result = true;
  if (std::abs(z) < lambda1) {
    *W = 0.0f;
    result = false;
  } else {
    *W = -1.0f / (lambda2 + (beta + sqrt(q)) / alpha) *
        (z - lambda1 * (z > 0.0f ? 1.0f : -1.0f));
  }
  return result;
}


static const int optBufSize = 3;

using std::array;

class Optimizer {
 public:
  virtual bool optProcess(array<valueType, optBufSize>* , valueType* ,
                          const valueType& g) = 0;
  virtual ~Optimizer() {}
};

class ftrlOptimizer : public Optimizer {
 public:
  ftrlOptimizer(valueType a = 0.1f, valueType b = 1.0f,
                valueType l1 = 0.4f, valueType l2 = 1.0f);
  bool optProcess(array<valueType, optBufSize>* buf,
                  valueType* W, const valueType& g);
 private:
  valueType alpha;
  valueType beta;
  valueType lambda1;
  valueType lambda2;
};
