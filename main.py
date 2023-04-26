import matplotlib.pyplot as plt
from k_means_functions import sk_kmeans, k_means, mini_batch_k_means, k_means_plusplus, kmeans_PCA_Part, show_main_colors, main

img = plt.imread('input_images/p1.jpg')
k = 3
max_iterations = 150
batch_size = 128

# 1.Sklearn的K-Means
# main_colors = sk_kmeans(img, k)

# 2.Sample K-Means
# main_colors = k_means(img, k, max_iterations)

# # 3.Mini Batch K_means
# main_colors = mini_batch_k_means(img, k, batch_size, max_iterations)

# 4.K-Means++
# main_colors = k_means_plusplus(img, k, max_iterations)

# 5.K-Means (PCA-Part)
main_colors = kmeans_PCA_Part(img, k)

show_main_colors(main_colors, k, to_int=True)

main(img, main_colors, k)