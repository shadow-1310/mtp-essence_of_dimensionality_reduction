import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
from tqdm import tqdm
import numpy as np
import cv2
from time import time
import random
from scipy.stats import kurtosis, kurtosistest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


RANDOM_STATE = 13

class InitializeData:
    def __init__(self, metadata_path, data_path) -> None:
        self.metadata_path = metadata_path
        self.data_dir = data_path
        self.df = self.initialize_metadata()
        self.image_data = self.read_image_data()

    def initialize_metadata(self):
        df = pd.read_csv(self.metadata_path)
        df.drop(columns=['Unnamed: 0'], inplace=True)
        return df

    def read_image_data(self):
        tick = time()
        img = self.df['image_filename'].apply(lambda x: cv2.imread(self.data_dir + x, 0))
        tock = time()
        print("Time required for converting the images to pixel array: ", tock - tick, "seconds") 
        image_data = np.stack(img, axis=0)
        image_data = image_data.reshape(5063, -1)
        return image_data


def make_path(path_type, path):
    if path_type == 'data':
        if not os.path.exists('data/'):
            os.makedirs('data')
        path_data = os.path.join('data', path)
        print(path_data)
        return path_data

    else:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        path_image = os.path.join('plots', path)
        print(path_image)
        return path_image

class GeneratePlots:
    def __init__(self, metadata, image_data) -> None:
        self.image_data = image_data
        self.df = metadata



    def generate_histogram(self, image_class, path_image):
        if image_class == 0:
            image_id = random.choice(list(self.df[self.df['class_number'] == 0].index.values))
            image_type = 'healthy'
        else:
            image_id = random.choice(list(self.df[self.df['class_number'] == 1].index.values))
            image_type = 'cancerous'

        print(image_id)
        path = make_path('plot', path_image)

        plt.figure()
        sns.histplot(self.image_data[image_id])
        plt.title(f"Pixel distributions of a random {image_type} image")
        plt.savefig(path, format="eps", dpi=1200)
        plt.show()
    

    def generate_boxplot_kurtosis(self, path_data, path_image):
        all_k_values = []

        for i in tqdm(range(len(self.image_data)), total = len(self.image_data)):
            all_k_values.append(kurtosis(self.image_data[i]))

        dic = {
            'image_name': self.df['image_filename'],
            'kurtois_value': all_k_values
        }
        path_image = make_path('plot', path_image)
        path_data = make_path('data', path_data)

        df_kurtosis = pd.DataFrame(dic)
        df_kurtosis.to_csv(path_data)

        plt.figure()
        sns.boxplot(all_k_values, orient='h')
        plt.savefig(path_image, format="eps", dpi=1200)
        plt.show()


    def generate_covariance_plot(self, path_image, zoom_level):
        start = time()
        pca = PCA(n_components = None)
        pca_components = pca.fit_transform(self.image_data)
        end = time()
        print("time taken for doing PCA: ", end -start, "seconds")

        path_image = make_path('plot', path_image)

        font_prop = fm.FontProperties(fname='times.ttf')
        if zoom_level:
            xlim = zoom_level[0]
            ylim = zoom_level[1]
            plt.grid()
            plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
            plt.xlim(xlim[0], xlim[1])
            plt.ylim(ylim[0], ylim[1])

        else:
            plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))

        plt.xlabel("Number of components", fontproperties=font_prop, fontsize=12)
        plt.ylabel("Explained Variance", fontproperties=font_prop, fontsize=12)
        plt.title("Perecentage of cumulative variance with components", y=1.03)
        plt.savefig(path_image, format='eps', dpi=1200)
        plt.show()
        plt.close()


class Generate2dVisualization:
    def __init__(self, metadata_path, data_path):
        i = InitializeData(metadata_path, data_path)
        self.df = i.df
        self.image_data = i.image_data

    def generate_scatterplot(self, data, path_data, path_image, title):
        font_prop = fm.FontProperties(fname='times.ttf')

        path_data = make_path('data', path_data)
        path_image = make_path('plot', path_image)

        dic = {
            'component1' : data[:, 0],
            'component2' : data[:, 1]
        }
        scatter_df = pd.DataFrame(dic)
        scatter_df.to_csv(path_data)

        ax = sns.scatterplot(x=data[:,0], y=data[:,1], hue = self.df['class_number'])
        plt.xlabel("Component 1", fontproperties = font_prop, fontsize = 14, weight = 'bold')
        plt.ylabel("Component 2", fontproperties = font_prop, fontsize = 14, weight = 'bold')
        plt.title(title, weight = 'bold', fontsize = 12)
        # ax.legend(prop = font_prop)
        plt.savefig(path_image, format='eps', dpi = 1200)
        plt.show()
        plt.close()


    def generate_pca(self, pca_variant, kernel,  path_data, path_image):
        if pca_variant == 'linear':
            start = time()
            scaler = StandardScaler()
            image_transformed = scaler.fit_transform(self.image_data)
            pca = PCA(n_components = 2)
            pca2 = pca.fit_transform(image_transformed)
            end = time()
            print("time taken for doing Linear PCA: ", end -start, "seconds")
            title = '2D representation of the images after using Linear PCA'
            self.generate_scatterplot(pca2, path_data, path_image, title)

        elif pca_variant == 'kernel':
            start = time()
            scaler = StandardScaler()
            image_transformed = scaler.fit_transform(self.image_data)
            kpca = KernelPCA(n_components = 2, random_state = RANDOM_STATE, kernel = kernel)
            kpca2 = kpca.fit_transform(image_transformed)
            end = time()
            print(f"time taken for doing kPCA with {kernel} kernel: ", end -start, "seconds")
            
            title = f'2D representation of the images after using kernel PCA ({kernel})'
            self.generate_scatterplot(kpca2, path_data, path_image, title)


    def generate_lda(self, path_data, path_image):
        start = time()
        lda = LinearDiscriminantAnalysis()
        image_lda = lda.fit_transform(self.image_data, self.df['class_number'])
        end = time()
        print("time taken for doing LDA: ", end -start, "seconds")

        path_data = make_path('data', path_data)
        path_image = make_path('plot', path_image)

        df_lda = pd.DataFrame({'componet1': image_lda[:, 0]})
        df_lda.to_csv(path_data, index=False)

        font_prop = fm.FontProperties(fname='times.ttf')
        ax = sns.scatterplot(x=image_lda[:,0], y=[0]*(image_lda.shape[0]), hue = self.df['class_number'])
        plt.xlabel("Component 1", fontproperties = font_prop, fontsize = 16)
        ax.legend(prop=font_prop)
        plt.title("One dimensional plot after using LDA")
        plt.savefig(path_image, format='eps', dpi=1200)
        plt.show()
    

    def generate_fa(self, path_data, path_image, svd, rotation):
        start = time()

        scaler = StandardScaler()
        image_transformed = scaler.fit_transform(self.image_data)

        fa = FactorAnalysis(random_state = RANDOM_STATE, n_components = 2, svd_method=svd, rotation=rotation)
        image_fa = fa.fit_transform(image_transformed)
        end = time()
        print("time taken for doing FA: ", end -start, "seconds")

        title = f'2D representation of the images after using FA (svd: {svd}, rotation: {rotation})'
        self.generate_scatterplot(image_fa, path_data, path_image, title)

