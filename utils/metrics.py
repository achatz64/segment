from tensorflow import shape, argmax, range, map_fn, where, zeros, cond, int32, float32, float64, reduce_mean, unique, reshape, constant

def precision_recall(y_true, y_pred):

  num_classes = shape(y_true)[-1]

  y_total_true = argmax(y_true, -1, int32)
  y_total_pred = argmax(y_pred, -1, int32)

  def precision_recall_per_class(category):

      restricting_to_class = where(y_total_true == category, y_total_pred, y_total_true)
      intersection = shape(where(restricting_to_class == category))[0]

      all_true_in_category = shape(where(y_total_true == category))[0]
      all_pred_in_category = shape(where(y_total_pred == category))[0]

      precision = cond(all_pred_in_category == 0, lambda: constant(1, dtype=float64), lambda: intersection/all_pred_in_category)
      recall = cond(all_true_in_category == 0, lambda: constant(1, dtype=float64), lambda: intersection/all_true_in_category)
      jaccard = cond(all_true_in_category + all_pred_in_category - intersection == 0, lambda: constant(1, dtype=float64),
                     lambda: intersection/(all_pred_in_category + all_true_in_category - intersection))

      return (precision, recall, jaccard)


  return map_fn(precision_recall_per_class, range(num_classes), dtype = (float64, float64, float64))

def mean_jaccard(y_true, y_pred):

  num_classes = shape(y_true)[-1]

  y_total_true = argmax(y_true, -1, int32)
  y_total_pred = argmax(y_pred, -1, int32)

  def precision_recall_per_class(category):

      restricting_to_class = where(y_total_true == category, y_total_pred, y_total_true)
      intersection = shape(where(restricting_to_class == category))[0]

      all_true_in_category = shape(where(y_total_true == category))[0]
      all_pred_in_category = shape(where(y_total_pred == category))[0]

      jaccard = cond(all_true_in_category + all_pred_in_category - intersection == 0, lambda: constant(1, dtype=float64),
                     lambda: intersection/(all_pred_in_category + all_true_in_category - intersection))

      return jaccard


  return reduce_mean(map_fn(precision_recall_per_class, range(num_classes), dtype = float64))

def effective_mean_jaccard(y_true, y_pred):

  y_total_true = argmax(y_true, -1, int32)
  y_total_pred = argmax(y_pred, -1, int32)

  true_classes, _ = unique(reshape(y_total_true, [-1]))

  def precision_recall_per_class(category):

      restricting_to_class = where(y_total_true == category, y_total_pred, y_total_true)
      intersection = shape(where(restricting_to_class == category))[0]

      all_true_in_category = shape(where(y_total_true == category))[0]
      all_pred_in_category = shape(where(y_total_pred == category))[0]

      jaccard = cond(all_true_in_category + all_pred_in_category - intersection == 0, lambda: constant(1, dtype=float64),
                     lambda: intersection/(all_pred_in_category + all_true_in_category - intersection))

      return jaccard

  return reduce_mean(map_fn(precision_recall_per_class, true_classes, dtype = float64))

def baseline_effective_jaccard(y_true, y_pred):

  y_total_true = argmax(y_true, -1, int32)
  y_total_pred = zeros(shape(y_total_true), int32)

  true_classes, _ = unique(reshape(y_total_true, [-1]))

  def precision_recall_per_class(category):

      restricting_to_class = where(y_total_true == category, y_total_pred, y_total_true)
      intersection = shape(where(restricting_to_class == category))[0]

      all_true_in_category = shape(where(y_total_true == category))[0]
      all_pred_in_category = shape(where(y_total_pred == category))[0]

      jaccard = cond(all_true_in_category + all_pred_in_category - intersection == 0, lambda: constant(1, dtype=float64),
                     lambda: intersection/(all_pred_in_category + all_true_in_category - intersection))

      return jaccard

  return reduce_mean(map_fn(precision_recall_per_class, true_classes, dtype=float64))

