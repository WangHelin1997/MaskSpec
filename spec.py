import librosa
import numpy as np
filename = librosa.ex('trumpet')
y, sr = librosa.load(filename, offset=0, duration=10.0)
print(sr, y.shape)
x = librosa.feature.melspectrogram(y=y, sr=sr)
x = librosa.power_to_db(x, ref=np.max)
print(x.shape) # 128, 431
x_ = x[:, :192]
print(x.shape) # 32*4, 32*6
import matplotlib.pyplot as plt
plt.matshow(x_)
plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
plt.axis('off')
plt.savefig('spec.pdf', dpi=300,bbox_inches='tight',pad_inches=0.0)
for i in range(4):
    for j in range(6):
        x = x_[32*i:32*(i+1),32*j:32*(j+1)]
        plt.matshow(x)
        plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
        plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
        plt.axis('off')
        plt.savefig('spec'+str(i)+str(j)+'.pdf', dpi=300,bbox_inches='tight',pad_inches=0.0)
