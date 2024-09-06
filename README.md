# BTT-Spring-2024-AI-Studio
Team Powderpuffs - 2nd Place Solution for the BTTAI x NYBG Spring 2024 AI Studio

## Context
Business context: https://www.kaggle.com/competitions/bttai-nybg-2024/overview <br/>
Data context: https://www.kaggle.com/competitions/bttai-nybg-2024/data

## Overview of the approach
Our final model was a combination (ensemble) of 8 high scoring models. Resnet101 (Public LB of 0.9816), Mobilenetv2 (0.9861), Inceptionv3 (0.9863), Nasnet (0.9876), Densenet121 (0.9876), Xception (0.9874), Vgg19 (0.9833), and finally Efficientnetb4 (0.9839). These models specifically were chosen due to their reputation of performing well for multi classification cases, and the pretrained (imagenet) versions being available for easy use on keras.

Our ensemble with these models was a simple model stacking approach, where we trained the base models individually (as seen above) and used different combinations of them to essentially select a meta model, which makes the final predictions. We feel that a full stacking approach, where a meta model is actually trained on the predictions would probably be better. But we were never able to find a good meta learner that acclimated well.

## Details of the submission
### Mixing
Augmentation was a difficult thing to approach; without it, the models would overfit, and with it, the models would perform lackluster. We tried simple rotations and other methods, but they were never up to par. We stumbled upon a technique called Mixup, which worked magnificently with the dataset:

Mixup is a data augmentation technique that blends two images and their labels to create a new synthetic sample. This new sample is a linear combination of the original images and labels, controlled by a mixing coefficient lambda (λ). This coefficient is drawn from a beta distribution with a parameter alpha, which controls the shape of the distribution.

The mixup_generator function below implements this technique. It takes a generator that yields batches of images and labels, and then, for each batch, it generates a new batch by blending two consecutive batches from the original generator. By using mixup, we were able to create more diverse training samples, which helped prevent overfitting and improved the generalization ability of our models.
```ruby
def mixup_generator(generator, alpha=0.2):
   while True:
       images_X1, labels_X1 = next(generator)
       images_X2, labels_X2 = next(generator)
       batch_size = min(images_X1.shape[0], images_X2.shape[0])
       lam = np.random.beta(alpha, alpha, size=batch_size)
       lam = lam.reshape(batch_size, 1, 1, 1)
       images = lam * images_X1[:batch_size] + (1 - lam) * images_X2[:batch_size]
       labels = lam.reshape(batch_size, 1) * labels_X1[:batch_size] + (1 - lam).reshape(batch_size, 1) * labels_X2[:batch_size]
       yield images, labels
```
### Magic optimizer
Our individual models were plateauing at a certain point. We tried different layers, dropouts, and feature filtering techniques, but it would end up reaching the same score—just from different directions.

Once this was realized, I personally thought it was time to pack up our bags and try a different approach. Lucy one day implemented the SGD optimizer, with specific parameters and the difference was significant. We were initially using Adam optimizer, as that is the standard—however, the model would always get stuck. Not in a way that was natural; more so in a janky way. But with SGD the momentum and Nesterov acceleration helped the models escape local minima more effectively and converge faster to a better solution. It seemed to smooth out the optimization process, making the training more stable and less prone to getting stuck.

### Leverage
We went in trying CNN, UCNN and other models. They weren’t terrible, but it didnt seem realistic to get a good score off of these models which trained only on our ‘small’ dataset. Thus we adopted transfer learning using different keras pretrained models on Imagenet. This allowed us to leverage the knowledge learned from a large dataset (ImageNet, 14,000,000 images) and apply it to our dataset for classification. From there fine tuning it for our use cases was seamless.

### Limit
Kaggle allots 30 hours of weekly gpu usage, which is quite generous. Though realistically it was pretty limiting in what we could do, especially since it would take roughly 12 hours to train a single model. This left much less than desired in what we could test. Though this did make us more efficient in our implementations, doing more thorough research before trying them out. This limited us to roughly 20 epochs for each, more potentially could’ve improved the model by a tiny amount, but would have to see with more runtime.

### SE
This was something we tried with our later individual models, where we added a Squeeze-and-Excitation (SE) block to attempt to enhance performance. The models' scores were roughly the same with or without the SE block, but we believe it added more variability to the models--which would require further testing.

The reason behind this block is that some parts of the pictures in the dataset were more important for understanding what's in them. Thus the SE block was intended to help the neural network pay more attention to the important parts and less attention to the less important parts. Though it didn't show any noticeable improvement in our case.
```ruby
def se_block(in_block, ch, ratio=16):
   y = GlobalAveragePooling2D()(in_block)
   y = Dense(ch // ratio, activation='relu')(y) 
   y = Dense(ch, activation='sigmoid')(y)       
   y = Reshape((1, 1, ch))(y)

   return multiply([in_block, y])
```
## Insights (Not Implemented in final models)
### Finetuning
The approach we had was to freeze a certain amount of layers from the original pre trained model. Then we train that model on our dataset at a standard learning rate. Afterward we would unfreeze those layers and train it at a much lower learning rate. This would make the model more acclimated to our dataset and see the subtleties. But it didnt provide any improvements that could be seen with our attempts. The pattern we saw was that a higher amount of layers frozen would reduce model score, so perhaps a lower one could improve.

### Ensemble
We tried to make a meta-model that is learned off of the predictions of our base models, but the score would always be awful. Could be a discrepancy in our understanding and implementation, but we believed that the meta learner we tried (LRR and RF) could not acclimate well to our complex predictions.

### LR
We tested several different learning rates, including 0.01, 0.001, and 0.0001 with 0.01 performing best for with our SGD optimizer.

## Example Individual Model Code
```ruby
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, multiply, Reshape
from tensorflow.keras.optimizers import SGD

train_data = pd.read_csv("/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-train.csv")
val_data = pd.read_csv("/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-validation.csv")
test_data = pd.read_csv("/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-test.csv")

train_dir = '/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-train/BTTAIxNYBG-train'
val_dir = '/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-validation/BTTAIxNYBG-validation'
test_dir = '/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-test/BTTAIxNYBG-test'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
train_generator = train_datagen.flow_from_dataframe(
   dataframe=train_data,
   directory=train_dir,
   x_col="imageFile",
   y_col="classLabel",
   target_size=(224, 224),
   batch_size=batch_size,
   class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
   dataframe=val_data,
   directory=val_dir,
   x_col="imageFile",
   y_col="classLabel",
   target_size=(224, 224), 
   batch_size=batch_size,
   class_mode='categorical'
)

def mixup_generator(generator, alpha=0.2):
   while True:
       images_X1, labels_X1 = next(generator)
       images_X2, labels_X2 = next(generator)
       batch_size = min(images_X1.shape[0], images_X2.shape[0])
       lam = np.random.beta(alpha, alpha, size=batch_size)
       lam = lam.reshape(batch_size, 1, 1, 1)

       images = lam * images_X1[:batch_size] + (1 - lam) * images_X2[:batch_size]
       labels = lam.reshape(batch_size, 1) * labels_X1[:batch_size] + (1 - lam).reshape(batch_size, 1) * labels_X2[:batch_size]

       yield images, labels

def se_block(in_block, ch, ratio=16):
   y = GlobalAveragePooling2D()(in_block)
   y = Dense(ch // ratio, activation='relu')(y) 
   y = Dense(ch, activation='sigmoid')(y)       
   y = Reshape((1, 1, ch))(y)

   return multiply([in_block, y])

base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
l2_reg = 0.001
x = base_model.output
x = se_block(x, 1792) 
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

train_generator_mixup = mixup_generator(train_generator)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

model_checkpoint = ModelCheckpoint('EfficientNetB4_best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, mode='min', verbose=1)

optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
   train_generator_mixup, 
   epochs=25, 
   validation_data=val_generator, 
   steps_per_epoch=len(train_generator), 
   validation_steps=len(val_generator), 
   callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
   dataframe=test_data,
   directory=test_dir,
   x_col="imageFile",
   y_col=None,
   target_size=(224, 224), 
   batch_size=batch_size,
   class_mode=None,
   shuffle=False
)

test_preds = model.predict(test_generator)
test_predictions = test_preds.argmax(axis=-1)

test_data['classID'] = test_predictions
test_data[['uniqueID', 'classID']].to_csv('predictions.csv', index=False)
```
## Ensemble Model Code
```ruby
from keras.models import load_model
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from itertools import combinations

model_resnet = load_model('/kaggle/input/topsevenmodels/resnet101_9816.h5')
model_mobilenet = load_model('/kaggle/input/topsevenmodels/mobilenetv2_9861.h5')
model_inception = load_model('/kaggle/input/topsevenmodels/inceptionv3_9863.h5')
model_nasnetmobile = load_model('/kaggle/input/topsevenmodels/nasnet_9876.h5')
model_densenet = load_model('/kaggle/input/topsevenmodels/densenet121_9876.h5')
model_xception = load_model('/kaggle/input/topsevenmodels/xception_9874.h5')
model_vgg = load_model('/kaggle/input/topsevenmodels/vgg19_9833.h5')
model_efficient = load_model('/kaggle/input/efficientnetb4-9839/efficientnetb4_9839.h5')

train_data = pd.read_csv("/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-train.csv")
val_data = pd.read_csv("/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-validation.csv")
test_data = pd.read_csv("/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-test.csv")

train_dir = '/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-train/BTTAIxNYBG-train'
val_dir = '/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-validation/BTTAIxNYBG-validation'
test_dir = '/kaggle/input/bttai-nybg-2024/BTTAIxNYBG-test/BTTAIxNYBG-test'

def ensemble_predictions(models, generator):
   predictions = [model.predict(generator, steps=len(generator)) for model in models]
   avg_preds = np.mean(predictions, axis=0)
   return avg_preds

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 64
train_generator = train_datagen.flow_from_dataframe(
   dataframe=train_data,
   directory=train_dir,
   x_col="imageFile",
   y_col="classLabel",
   target_size=(224, 224),
   batch_size=batch_size,
   class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
   dataframe=val_data,
   directory=val_dir,
   x_col="imageFile",
   y_col="classLabel",
   target_size=(224, 224),
   batch_size=batch_size,
   class_mode='categorical'
)

candidate_models = [model_resnet, model_mobilenet, model_inception, model_nasnetmobile, model_densenet, model_xception, model_vgg, model_efficient]

best_accuracy = 0
best_ensemble = None

ensemble_combinations = combinations(candidate_models, 6)

for models_combination in ensemble_combinations:
   models = list(models_combination)

   val_preds = ensemble_predictions(models, val_generator)
   val_labels = np.argmax(val_preds, axis=1)
   val_accuracy = accuracy_score(val_generator.classes, val_labels)

   if val_accuracy > best_accuracy:
       best_accuracy = val_accuracy
       best_ensemble = models

print(f'Best Validation Accuracy: {best_accuracy}')
print(f'Best Ensemble Models: {best_ensemble}')

final_val_preds = ensemble_predictions(best_ensemble, val_generator)
final_val_labels = np.argmax(final_val_preds, axis=1)
final_val_accuracy = accuracy_score(val_generator.classes, final_val_labels)
print(f'Final Validation Accuracy with Best Ensemble: {final_val_accuracy}')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
   dataframe=test_data,
   directory=test_dir,
   x_col="imageFile",
   y_col=None,
   target_size=(224, 224),
   batch_size=batch_size,
   class_mode=None,
   shuffle=False
)
test_preds = ensemble_predictions(best_ensemble, test_generator)
test_labels = np.argmax(test_preds, axis=1)

test_data['classID'] = test_labels
test_data[['uniqueID', 'classID']].to_csv('ensemble_predictions.csv', index=False)
```
## Acknowledgments

This project was completed with the collaborative effort of the following team members:
- Garv Sehgal
- Hadassah Krigsman
- Jing Gan

I want to express my sincere gratitude for their invaluable contributions to the project.
