
# XGBoost ML model for Grade 02/23/23 -------------------------------------

library(tidyverse)
library(tidymodels)
library(here)
library(patchwork)


# data load ---------------------------------------------------------------

chem1 <-
  read_csv("C:\\Users\\Joel\\OneDrive - RSI Pipeline Solutions\\ChemistryGrade\\all_data.csv") %>%
  filter(wall > 0) %>%
  select(
    c,
    mn,
    s,
    #including Sulfur only increased accuracy 0.1% when using ys
    OD,
    wall,
    grade,
    ys,
    # uts,
    # vintage
  ) %>%
  mutate(
    DT = OD / wall,
    grade = factor(grade),
    s = if_else(s>0.1, s/10,s),
    # R = ys/uts
  ) %>%
  drop_na()

Early <- chem1 %>%
  filter(grade=="Early")

B <- chem1 %>%
  filter(grade=="Grade B", ys>=35)

x42 <- chem1 %>%
  filter(grade=="X42", ys>=41)

x46 <- chem1 %>%
  filter(grade=="X46", ys>=45)

x52 <- chem1 %>%
  filter(grade=="X52", ys>=51)

x60 <- chem1 %>%
  filter(grade=="X60", ys>=59)

x65 <- chem1 %>%
  filter(grade=="X65", ys>=64)

x70 <- chem1 %>%
  filter(grade=="X70", ys>=69)

chem <- bind_rows(Early, B, x42, x46, x52, x60, x65, x70) %>%
  select(-ys)

chem_names <- names(chem)

# Train/Test Split --------------------------------------------------------

trainSplit <- initial_split(chem, strata=grade, prop = 0.85)
trainData <- training(trainSplit)
testData <-  testing(trainSplit)

# Do Parallel -------------------------------------------------------------

all_cores <- parallel::detectCores(logical = FALSE)
doParallel::registerDoParallel(all_cores)

# Recipe ------------------------------------------------------------------

grade_recipe <-
  recipes::recipe(grade ~ . , data = trainData) %>%
  # convert categorical variables to factors
  recipes::step_string2factor(all_nominal()) %>%
  # # combine low frequency factor levels
  # recipes::step_other(all_nominal(), threshold = 0.01) %>%
  # # remove no variance predictors which provide no predictive information
  prep()

train_processed <- bake(grade_recipe,  new_data = training(trainSplit))

# Cross Validation --------------------------------------------------------

grade_cv_folds <-
  recipes::bake(grade_recipe,
                new_data = trainData) %>%
  rsample::vfold_cv(v = 10, strata = grade)

# Model Spec --------------------------------------------------------------

xgboost_model <-
  parsnip::boost_tree(
    mode = "classification",
    trees = 450,
    min_n = 1,
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune()
  ) %>%
  set_engine("xgboost") %>%
  translate()

xgboost_model

# Grid Spec ---------------------------------------------------------------

xgboost_params <-
  dials::parameters(
    # min_n(), # set min_n to 1 in model definition
    tree_depth(),
    learn_rate(),
    loss_reduction()
  )

# Boost Grid #################
xgboost_grid <-
  dials::grid_max_entropy(xgboost_params,
                          size = 30)

# Workflow ----------------------------------------------------------------
# We use the new tidymodels workflows package to add a formula to our XGBoost
# model specification.

xgboost_wf <-
  workflows::workflow() %>%
  add_model(xgboost_model) %>%
  add_formula(grade ~ .)

# Hyperparameter Tuning ---------------------------------------------------

# Comment this out if recalling saved parameters
# xgboost_tuned <- tune::tune_grid(
#   object = xgboost_wf,
#   resamples = grade_cv_folds,
#   grid = xgboost_grid,
#   metrics = yardstick::metric_set(accuracy),
#   control = tune::control_grid(verbose = TRUE)
# )

# Finalize Workflow -------------------------------------------------------

# Comment this out if using saved parameters
# xgboost_tuned %>%
#   tune::show_best(metric = "accuracy") %>%
#   knitr::kable()

# Next, isolate the best performing hyperparameter values.
# xgboost_best_params <- xgboost_tuned %>%
#   tune::select_best("accuracy");xgboost_best_params
#
# write_csv(xgboost_best_params, "bestparams_R_ratio_DT.csv")

# xgboost_best_params <- read_csv("bestparams_R_ratio_DT.csv")
xgboost_best_params <- read_csv("best_params_no_tensile_mtry.csv")


# final model is used for new predictions
xgboost_model_final <- xgboost_model %>%
  finalize_model(xgboost_best_params)

# final workflow used for CM and predictions
xgboost_model_wf <- xgboost_wf %>%
  finalize_workflow(xgboost_best_params)


# Fit Model and Collect Metrics ---------------------------------------------

model_fit <- xgboost_model_final %>%
  # fit the model on all the training data
  fit(
    formula = grade ~ .,
    data    = train_processed
  )


final_res <- last_fit(xgboost_model_wf,trainSplit)

collect_metrics(final_res)

test_predicitons <- final_res %>%
  collect_predictions()

# Confusion Matrix --------------------------------------------------------

cm <- test_predicitons %>%
  conf_mat(truth =grade,estimate =.pred_class);cm

# Variable Importance -----------------------------------------------------

model_fit %>%
  vip::vip(aesthetics = list(fill = 'steelblue')) +
  theme_bw()


# New Data ---------------------------------------------------------

## Color Spec for plots

pal <-
  c(
    "#e6194B",
    "#f58231",
    "#ffe119",
    "#3cb44b",
    "#42d4f4",
    "#4363d8",
    "#911eb4",
    "#f032e6"
  )

grd_lvl <- c("LTB", "B", "X42", "X46", "X52", "X60", "X65", "X70")


new_dat <-
  read_csv (
    "C:\\Users\\Joel\\OneDrive - RSI Pipeline Solutions\\ChemistryGrade\\stations\\Gosford\\data\\GosfordRd_inputs_All_NWT1-OD1_2023-02-16.csv"
  ) %>%
  janitor::clean_names() %>%
  rename(
    ys = yield_strength,
    uts = tensile_strength,
    wall = pipe_wt,
    OD = pipe_od
    ) %>%
  mutate(
    ys = 1.0 * ys,
    R = ys / uts,
    DT = OD / wall #,
    # vendor = "Aud"
  ) %>%
  filter(!is.na(R))





# New Predictions ---------------------------------------------------------

fit_new_dat <-  bind_cols(new_dat,predict(model_fit, new_dat, type="prob")) %>%
  mutate(model="Chem only",
         # scenario = "c = 1.0"
         ) %>%
  rename(.pred_LTB = .pred_Early,
         .pred_B = '.pred_Grade B')

# new_dat1 = fit_new_dat
# fit_new_dat_c <- bind_rows(new_dat0, new_dat1)


# Write Predictions -------------------------------------------------------

write_csv(fit_new_dat, "C:\\Users\\Joel\\OneDrive - RSI Pipeline Solutions\\ChemistryGrade\\stations\\Gosford\\data\\gosferd_preds_02272023.csv")

# Mean Probability by Grade ------------------------------------------------

fit_new_dat %>%
  # filter(chemistry=="LIBS_XRF") %>%
  pivot_longer(.pred_LTB:.pred_X70) %>%
  mutate(name = str_remove(string = name,
                           pattern = ".pred_"),
         name = factor(name, levels = grd_lvl)
  ) %>%
  # ungroup() %>%
  group_by(feature, chemistry, name) %>%
  summarise(sum_tot = mean(value)) %>%
  ggplot(aes(name, sum_tot)) +
  geom_col(aes(fill = name),
           position = "dodge",
           show.legend = F) +
  scale_y_continuous(breaks = scales::pretty_breaks())+
  facet_wrap( feature ~ chemistry ,
              ncol=4) +
  theme_bw(16) +
  scale_fill_manual(values = pal) +
  labs(
    title = "Gosferd Mean Probabilities by Grade",
    # subtitle = "Using Different YS",+
    y = "Mean Probability",
    x = "Grade",
    caption = "Chem + Dt Model"
  )+
  theme(plot.margin = margin(0.7,0.7,0.7,0.7,"cm"),
        panel.spacing.x = unit(0.7,"cm") )

# ridgeline plot ----------------------------------------------------------

fit_new_dat %>%
  filter(chemistry == "Filings") %>%
  pivot_longer(.pred_LTB:.pred_X70) %>%
  mutate(
    name = str_remove(string = name,
                      pattern = ".pred_"),
    name = factor(name, levels = grd_lvl)
  ) %>%
  filter(name %in% c("LTB", "B", "X42", "X52")) %>%
  ggplot(aes(value)) +
  ggridges::geom_density_ridges(aes(y = scenario, fill = scenario),
                                show.legend = FALSE,
                                alpha = 0.5) +
  facet_wrap(feature ~ name)+
  theme_bw()
