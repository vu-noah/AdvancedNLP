# 28.02.2023

import preprocess_datasets
import logistic_regression_model
import step1_feature_extraction_candidates
import step2_feature_extraction_categories


def main():
    preprocess_datasets.preprocess_dataset('../Data/en_ewt-up-train.conllu')
    preprocess_datasets.preprocess_dataset('../Data/en_ewt-up-test.conllu')

    df_train, candidate_cat_feature_dicts_train, candidate_num_feature_dicts_train = \
        step1_feature_extraction_candidates.extract_features_to_determine_candidates('Data/train_data_only_current_candidates.tsv')
    df_test, candidate_cat_feature_dicts_test, candidate_num_feature_dicts_test = \
        step1_feature_extraction_candidates.extract_features_to_determine_candidates('Data/test_data_only_current_candidates.tsv')

    logistic_regression_model.run_logreg(candidate_cat_feature_dicts_train, candidate_num_feature_dicts_train,
                                         df_train['is_candidate'].tolist(), candidate_cat_feature_dicts_test,
                                         candidate_num_feature_dicts_test, df_test['is_candidate'].tolist(),
                                         df_test, 'candidates')

    df_train, role_cat_feature_dicts_train, role_num_feature_dicts_train = \
        step2_feature_extraction_categories.extract_features_to_determine_roles('Data/train_data_only_current_candidates.tsv')
    df_test, role_cat_feature_dicts_test, role_num_feature_dicts_test = \
        step2_feature_extraction_categories.extract_features_to_determine_roles('Data/test_data_with_candidate_predictions.tsv')

    logistic_regression_model.run_logreg(role_cat_feature_dicts_train, role_num_feature_dicts_train,
                                         [role for role in df_train['semantic_role'] if role != 'V' and role != '_'],
                                         role_cat_feature_dicts_test, role_num_feature_dicts_test,
                                         [role for i, role in enumerate(df_test['semantic_role']) if
                                          df_test['candidate_prediction'][i] == 1], df_test, 'roles')


if __name__ == '__main__':
    main()
