import tensorflow
from matplotlib import pyplot as plt
import cv2
import numpy as np

from keras.applications.vgg16 import preprocess_input

#önceden eğitilmiş bir cnn modelini (vgg16) kullanacağız. (transfer öğrenme denilen olay)
#vgg16 ile verilerden özellik çıkarımı yapmak için aşağıdaki kalıp kullanılıyor.
OzellikCikarmaFasli = tensorflow.keras.applications.VGG16(weights='imagenet', 
                                                          include_top=False, 
                                                          input_shape=(224, 224, 3))

#modeli, katmanları falan görmek için .summary kullanıyoruz. 
OzellikCikarmaFasli.summary()

#vgg16 modelindeki bazı katmanları donduracağız. çünkü çok fazla trainable parametre var. bunlar zaten eğitilmiş tekrar işleme sokmak yorar
#'block5_conv1' den öncesini donduruyor, durduruyoruz.
OzellikCikarmaFasli.trainable = True
set_trainable = False
for layer in OzellikCikarmaFasli.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

#boş bir model oluşturuyorum.
model = tensorflow.keras.models.Sequential()

#boş modelimin ilk katmanına üstteki vgg16'yı ekliyorum.
model.add(OzellikCikarmaFasli)

#vgg16 ile oluşturulan matrisleri tek boyutlu bir vektöre dönüştürüyorum (düzleştiriyorum -flat-).
model.add(tensorflow.keras.layers.Flatten())

#vgg'nin son  katmanında 7*7*512=25088 nöron vardı. ondan sonra bir 256 nöron, sınıflandırıcı daha ekledik.
model.add(tensorflow.keras.layers.Dense(256, activation='relu'))
#2 sınıf üzerinde çalışacağım için (bulutlu ve güneşli) devamına 2 nöron daha ekledim. akt. fonk. softmax
model.add(tensorflow.keras.layers.Dense(2, activation='softmax'))

#modeli compile ediyorum, sonlandırıyorum. tamamladım.
model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

#oluşturduğum modele bir bakıyorum
model.summary()

#####################################################

# modeli oluşturdum artık sıra verileri vermekte
train_dizini = 'egitim'
gecerleme_dizini = 'gecerleme'
test_dizini = 'test'

#veri ön işleme kısmı. verileri farklı yöntemlerle çoğaltabiliyoruz.
train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255, # piksel değerleri 0-255'den 0-1 arasına getiriliyor.
      rotation_range=40, # istenilen artırma işlemleri yapılabilir.
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
      )

train_generator = train_datagen.flow_from_directory(
        train_dizini,
        target_size=(224, 224),   #verilerin hepsini bu boyutta al
        batch_size=20,            #verileri 20şer 20şer oku
        class_mode='categorical',
        shuffle=True
        ) #eğitim verilerini çoğalttık. 


validation_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        ) #geçerleme verilerini çoğaltmaya gerek yok sadece ölçekledik

validation_generator = validation_datagen.flow_from_directory(
        gecerleme_dizini,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical',
        shuffle=True
        )  


# modeli oluşturduk, veriyi hazırladık. şimdi egitim kısmı.
egitimtakibi = model.fit_generator(
      train_generator,           #egitim verisi
      steps_per_epoch=10,       #her epochta 10 kez 16 adet veri gönderiyorsun
      epochs=2,                  # iki tane epoch olsun
      validation_data=validation_generator,    #geçerleme verisi
      validation_steps=1)        

#acc = egitim doggruluğu ---- val_acc= geçerleme dogruluğu

# modelimizi kaydediyoruz daha sonra kullanmak üzere, .h5 uzantısıyla kaydedilir.
model.save('havadurumu_modeli.h5')

# şimdi test edelim
test_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

test_generator = test_datagen.flow_from_directory(
        test_dizini,
        target_size=(224, 224),
        batch_size=20,
        )

#sonuçları yazdırıyoruz
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
