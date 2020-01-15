import os
import sys

################################################ EXAMPLE A1 ################################################
import A1.gender
import A1.landmarks_test as l1
import A1.landmarks as l2

tr_X, te_X, tr_Y, te_Y = gender.preprocess(l2)
tr_X, tr_Y, rbf_SVC = gender.train(tr_X, te_X, tr_Y, te_Y)
acc_A1_train = rbf_SVC.score(tr_X, tr_Y)
acc_A1_test = gender.testResults(l1)

# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task A2
import A2.smiling
import A2.landmarks_v2_test as l1
import A2.landmarks_v2 as l2

tr_X, te_X, tr_Y, te_Y = smiling.preprocess(l2)
tr_X, tr_Y, rbf_SVC = smiling.train(tr_X, te_X, tr_Y, te_Y)
acc_A2_train = rbf_SVC.score(tr_X, tr_Y)
acc_A2_test = smiling.testResults(l1)
# Clean up memory/GPU etc...


# ======================================================================================================================
dir_ = '../Dataset_original_AMLS_19-20/'

# Task B1
import B1.face_shape

train_generator, validation_generator, data_generator, df2, img2 = face_shape.preprocess(dir_ + 'cartoon_set/', dir_ + 'cartoon_set_test/')
# create model
model_B2 = face_shape.CNN_modelling()
# fit data into model
face_shape.fitDataInModel(model_B2, train_generator, validation_generator)
# evaluation
acc_B2_train, acc_B2_test = face_shape.evaluate(model_B2, data_generator, df2, img2)
# Clean up memory/GPU etc...


# ======================================================================================================================
# Task B2
import B2.eye_color

train_generator, validation_generator, data_generator, df2, img2 = eye_color.preprocess(dir_ + 'cartoon_set/', dir_ + 'cartoon_set_test/')
# create model
model_B2 = eye_color.CNN_modelling()
# fit data into model
eye_color.fitDataInModel(model_B2, train_generator, validation_generator)
# evaluation
acc_B2_train, acc_B2_test = eye_color.evaluate(model_B2, data_generator, df2, img2)
# Clean up memory/GPU etc...


# ======================================================================================================================
## Results:
tr_sc = saved_model.evaluate_generator(train_generator, steps = validation_generator.samples // 32, verbose=1)
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

