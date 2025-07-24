# Best model estimator
import itertools
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedKFold, learning_curve, validation_curve
import pandas as pd

def FindBestTuningModel(model, 
                        method       = "GridSearchCV", 
                        scoring      = "accuracy", 
                        random_state = 40020409, 
                        refit        = "r2",
                        n_jobs       = 1,
                        param_grid   = None, 
                        train_X      = None, 
                        train_y      = None):
    """
    Perform hyperparameter tuning for a given model using GridSearchCV or RandomizedSearchCV.

    This function performs a cross-validated hyperparameter search using either a full grid search
    (GridSearchCV) or a randomized parameter search (RandomizedSearchCV) with a repeated K-Fold strategy.
    It prints the best hyperparameters and the associated performance statistics.

    Args:
        model (estimator): The machine learning model to be tuned (e.g., RandomForestClassifier()).
        method (str, optional): The search method to use. Must be either "GridSearchCV" (default)
                                or "RandomizedSearchCV".
        scoring (str, optional): Scoring metric to evaluate the models. Default is "accuracy".
        random_state (int, optional): Random seed for reproducibility. Used only in RandomizedSearchCV.
        param_grid (dict): Dictionary of parameter grid (for GridSearchCV) or parameter distributions
                           (for RandomizedSearchCV).
        train_X (array-like): Feature matrix for training.
        train_y (array-like): Target vector corresponding to `train_X`.

    Returns:
        estimator: The best estimator fitted with the best found hyperparameters.

    Raises:
        ValueError: If `method` is neither "GridSearchCV" nor "RandomizedSearchCV".

    Prints:
        - Best hyperparameters found.
        - Mean and standard deviation of cross-validated accuracy for the best model.
    """
    
    from sklearn.model_selection import (GridSearchCV, 
                                         RandomizedSearchCV, 
                                         RepeatedKFold)
    import pandas as pd
    import numpy  as np

    cv = RepeatedKFold(n_splits     = 10, 
                       n_repeats    = 10, 
                       random_state = 40020409)

    grid_cv = None
    if method == "GridSearchCV":
        grid_cv = GridSearchCV(estimator  = model,
                               param_grid = param_grid,
                               scoring    = scoring,
                               cv         = cv,
                               n_jobs     = n_jobs,
                               verbose    = 3,
                               refit      = refit)
    elif method == "RandomizedSearchCV":
        grid_cv = RandomizedSearchCV(estimator           = model,
                                     param_distributions = param_grid,
                                     scoring             = scoring,
                                     cv                  = cv,
                                     n_iter              = 50,              # Số lượng mẫu thử nghiệm từ không gian tham số
                                     n_jobs              = n_jobs,
                                     verbose             = 3,
                                     refit               = refit,
                                     random_state        = random_state)
    else:
        raise ValueError(f"Unsupported search method: {method}")

    grid_cv.fit(train_X, train_y)

    print(f"\n{type(model).__name__}'s best parameters using {method}: {grid_cv.best_params_}")

    # Extract all test scores of best index
    cv_results = pd.DataFrame(grid_cv.cv_results_)
    split_cols = cv_results.filter(like="split").loc[grid_cv.best_index_]

    mean_acc = split_cols.mean()
    std_acc  = split_cols.std()

    print(f"Mean accuracy          : {mean_acc:.4f}")
    print(f"Accuracy std deviation : {std_acc:.4f}")

    return grid_cv.best_estimator_


# Model tuning
def plot_VC(X_data, Y_data, model, param_grid, n_jobs, ylim=None, log=True):
    for param_name, param_range in param_grid.items():
        train_scores, test_scores = validation_curve(
            model, X_data, Y_data,
            param_name  = param_name,
            param_range = param_range,
            scoring     = "accuracy",
            n_jobs      = n_jobs
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std  = np.std(train_scores, axis=1)
        test_scores_mean  = np.mean(test_scores, axis=1)
        test_scores_std   = np.std(test_scores, axis=1)

        # plt.figure(figsize=(8, 5))
        plt.title(f"Validation Curve for {param_name}")

        if ylim is not None:
            plt.ylim(*ylim)

        best_index       = np.argmax(test_scores_mean)
        best_param_value = param_range[best_index]
        best_score       = test_scores_mean[best_index]

        # 👉 Vẽ đường thẳng xanh nét đứt trước tiên
        plt.axvline(x=best_param_value, color='b', linestyle='--', label=f"Best Score = {best_score:.2f}")

        plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

        if log:
            plt.semilogx(param_range, train_scores_mean, 'o-', color="r", label="Training score")
            plt.semilogx(param_range, test_scores_mean, 'o-', color="g", label="Test score")
        else:
            plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Test score")

        y_text = best_score + 0.02 if ylim is None else min(best_score + 0.02, ylim[1] - 0.01)
        plt.text(best_param_value, y_text, f" Score: {best_score:.4f}\n {param_name}: {best_param_value}", 
                 color='b', fontsize=9, ha='left')

        plt.xlabel(f'Parameter: {param_name}')
        plt.ylabel('Score')
        plt.legend(loc="lower right")
        plt.grid()
        # plt.tight_layout()
        plt.show()

        # In kết quả ra ngoài terminal để tiện copy-paste
        print(f"Validation Curve for {param_name}:")
        print(f"  → Best {param_name} = {best_param_value}")
        print(f"  → Best CV Score     = {best_score:.4f}\n")
def plot_LC(X_data, Y_data, model, train_sizes, random_state, n_jobs):
    # plt.figure(figsize=(8, 5))
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    train_sizes, train_scores, test_scores = learning_curve(
        model, X_data, Y_data, n_jobs=n_jobs, train_sizes=train_sizes, random_state=random_state
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)

    # === Tìm vị trí có test score cao nhất
    best_index = np.argmax(test_scores_mean)
    best_train_size = train_sizes[best_index]
    best_score = test_scores_mean[best_index]

    # Tính tỉ lệ train size đó trên tổng dữ liệu
    best_train_ratio = best_train_size / len(X_data)

    # Vẽ đường thẳng nét đứt tại điểm đó (vẽ trước để không đè lên line)
    plt.axvline(x=best_train_size, color='b', linestyle='--', label=f"Best CV Score = {best_score:.2f}")

    # Vẽ vùng sai số
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean  - test_scores_std,  test_scores_mean  + test_scores_std,  alpha=0.1, color="g")

    # Vẽ đường train và test score
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean,  'o-', color="g", label="Test score")

    # Chú thích giá trị điểm số và train size tại vị trí đó
    plt.text(best_train_size + (train_sizes.max() * 0.02), best_score,
             f"Score: {best_score:.4f}\nTrain size: {int(best_train_size)}\nRatio: {best_train_ratio:.4f}",
             color='b', fontsize=9, ha='left')

    plt.legend(loc="best")
    # plt.tight_layout()
    plt.show()

    # In kết quả ra ngoài terminal để tiện copy-paste
    print("Learning Curve Results:")
    print(f"  → Best train size   = {best_train_size} samples")
    print(f"  → Ratio             = {best_train_ratio:.4f}")
    print(f"  → Best CV Score     = {best_score:.4f}\n")
def plot_LC_multi_models(X_datasets, Y_data, models, model_names, train_sizes, random_state, n_jobs):
    plt.figure(figsize=(12, 7))
    plt.title("Learning Curves for Multiple Models")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    # Danh sách màu khác nhau
    colors = itertools.cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
                              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])

    # Lưu trung bình test score của từng model để tính vị trí tốt nhất
    all_test_scores_mean = []

    for X_data, model, name in zip(X_datasets, models, model_names):
        color = next(colors)

        train_sizes_res, train_scores, test_scores = learning_curve(
            model, X_data, Y_data, n_jobs=n_jobs, train_sizes=train_sizes, random_state=random_state
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std  = np.std(train_scores, axis=1)
        test_scores_mean  = np.mean(test_scores, axis=1)
        test_scores_std   = np.std(test_scores, axis=1)

        all_test_scores_mean.append(test_scores_mean)

        # Đường train score - nét liền
        plt.plot(train_sizes_res, train_scores_mean, 'o-', color=color, label=f"{name} Train Score")
        plt.fill_between(train_sizes_res,
                         train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std,
                         alpha=0.1, color=color)

        # Đường CV score - nét đứt
        plt.plot(train_sizes_res, test_scores_mean, 'o--', color=color, label=f"{name} CV Score")
        plt.fill_between(train_sizes_res,
                         test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std,
                         alpha=0.1, color=color)

    # Tính trung bình test score tất cả models
    avg_test_scores = np.mean(all_test_scores_mean, axis=0)

    # Tìm train size tốt nhất
    best_index = np.argmax(avg_test_scores)
    best_train_size = train_sizes_res[best_index]
    best_score = avg_test_scores[best_index]

    # Tính tỷ lệ train_size / max_train_size
    train_size_ratio = best_train_size / max(train_sizes_res)

    # Vẽ đường thẳng tại train size tốt nhất
    plt.axvline(x=best_train_size, color='blue', linestyle='--', linewidth=1.5,
                label=f"Best avg CV Score = {best_score:.2f} at {best_train_size} ({train_size_ratio:.2%})")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    print(f"\n=> Best average CV Score = {best_score:.4f} at train size = {best_train_size} ({train_size_ratio:.4f})")