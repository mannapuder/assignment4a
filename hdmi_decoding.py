# Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from matplotlib import colors

# Constants and variables

fps = 60
screen_size = [640, 480]  # size of the display
real_size = [800, 525]  # size of the display with blanking time periods

f_p = (25.2 + 0.000099) * 1e6  # pixel-clock frequency
f_h = f_p / real_size[0]  # line rate
f_v = f_h / real_size[1]  # frame rate
f_c = 400 * 1e6  # centre frequency
f_s = 64 * 1e6  # sampling frequency

m = 8
f_r = m * f_p  # new sampling rate
spf = int(f_r / f_v)  # samples per frame
start = 855 * m
f_u = 60798440

starts = {'225': 250*m*800 + 650, '250': -10 * m + 800 * m * 145, '275': 40 * m + 800 * m * 45, '300': 280 * m + 800 * m * 40,
          '325': 570 * m + 375 * 800 * m, '350': 180 * m + 800 * m * 155, '375': 800 * m * 500 + 200 * m,
          '400': 855 * m, '425': 210050 * m, '450': 220 * m + 800 * m * 50, '475': 320 * m + 800 * m * 60}
f_u_values = {'225': 62199085, '250': 61998985, '275': 61798892, '300': 22798907, '325': 61398710, '350': 61198617,
              '375': 60998525, '400': 60798440, '425': 21798445, '450': 21598360, '475': 60198178}


# Load image

def load_data(filename):
    data = np.fromfile(filename, np.complex64)
    return data


def resampling(data, old, new):
    samples = int(len(data) * (new / old))
    resampled = resample(data, samples, window="parzen")
    return resampled


def show_img(frame):
    image = np.zeros((real_size[1], real_size[0]))
    pixel = 0
    for x in range(real_size[1]):
        for y in range(real_size[0]):
            sum = 0
            for i in range(m):
                sum += frame[pixel]
                pixel += 1
            sum /= m
            image[x][y] = sum
    plt.imshow(image[0:525, 0:800], cmap='gray')
    plt.title("Phase unrotation, " + mhz + "MHz")
    plt.show()


def show_img_colour(frame):
    image = np.zeros((real_size[1], real_size[0], 3))
    pixel = 0
    for x in range(real_size[1]):
        for y in range(real_size[0]):
            sum = 0
            for i in range(m):
                sum += frame[pixel]
                pixel += 1
            sum /= m
            image[x][y] = sum
    plt.imshow(image[0:525, 0:800])
    plt.title("Phase unrotation, " + mhz + "MHz")
    plt.show()


def half_top_half_bottom():
    data = get_resampled_data("data/scene3-640x480-60-400M-64M-40M.dat")
    first_frame = abs(np.array(data[start:int(start + 0.5 * spf)]))
    frame_10 = abs(np.array(data[int(start + 9.5 * spf):int(start + spf * 10) + 100]))
    conc = np.concatenate((first_frame, frame_10))
    show_img(conc)


def get_resampled_data(filename="data/scene3-640x480-60-400M-64M-40M.dat"):
    data = load_data(filename)
    resampled = resampling(data, f_s, f_r)
    return resampled


def demodulated_data_example(filename="data/scene3-640x480-60-400M-64M-40M.dat"):
    """
        Creates an example image of the display using data demodulation and one frame
    """
    data = get_resampled_data(filename)
    single_frame = np.array(data[start:start + spf])
    single_frame = abs(single_frame)
    show_img(single_frame)


def demodulation_frames_average_example():
    data = get_resampled_data("data/scene3-640x480-60-400M-64M-40M.dat")

    frames = 59
    n_frames = [abs(np.array(data[start + i * spf:start + (i + 1) * spf])) for i in range(frames)]
    averages = np.zeros(spf)
    for i in range(spf):
        averages[i] = sum(n_frames[j][i] for j in range(frames)) / frames

    show_img(averages)


def hsv_img_without_unrotation_example():
    data = load_data("data/scene3-640x480-60-400M-64M-40M.dat")

    frame = np.array(resampling(data, f_s, f_r)[start:start + spf])

    value = np.abs(frame)
    value *= 1 / value.max()

    hue = np.pi + np.angle(frame)
    hue *= 1 / hue.max()

    saturation = np.ones(hue.shape)

    hsv_image = np.dstack((hue, saturation, value))
    image = colors.hsv_to_rgb(hsv_image)[0]
    show_img_colour(image)


def hsv_img_with_unrotation_example(filename):
    data = load_data(filename)
    unrot = unrotate(data)

    frame = resampling(unrot, f_s, f_r)[start:]

    value = np.abs(frame)
    value *= 1 / value.max()

    hue = np.pi + np.angle(frame)
    hue *= 1 / hue.max()

    saturation = np.ones(hue.shape)

    hsv_image = np.dstack((hue, saturation, value))
    image = colors.hsv_to_rgb(hsv_image)[0]
    show_img_colour(image)


def hsv_img_with_unrotation_example_1frame():
    data = load_data("data/scene3-640x480-60-400M-64M-40M.dat")

    unrot = unrotate(data)

    frame = resampling(unrot, f_s, f_r)[start:start + fps]

    value = np.abs(frame)
    value *= 1 / value.max()

    hue = np.pi + np.angle(frame)
    hue *= 1 / hue.max()

    saturation = np.ones(hue.shape)

    hsv_image = np.dstack((hue, saturation, value))
    image = colors.hsv_to_rgb(hsv_image)[0]
    show_img_colour(image)


def spectrogram(samples):
    plt.specgram(samples, Fs=f_s)
    plt.axis(ymin=-4e7, ymax=4e7)
    plt.show()


def unrotate(data):
    return [data[i] * np.exp(2 * np.pi * 1j * i * f_u / f_s) for i in range(len(data))]


def separate_complex(data):
    real = np.isreal(data)
    imag = np.iscomplex(data)
    return data[real], data[imag]


def cartesian2polar(data):
    return [np.abs(data), np.angle(data)]


def add_noise(data_file, noise_file, distance):
    data = load_data(data_file)
    noise = load_data(noise_file)
    return (1 / distance ** 2) * data + noise


def demodulation_example_with_noise(noise_percentage):
    data = add_noise("data/scene3-640x480-60-400M-64M-40M.dat", "data/noise-640x480-60-400M-64M-40M.dat",
                     noise_percentage)
    data = resampling(data, f_s, f_r)
    frames = 59
    n_frames = [abs(np.array(data[start + i * spf:start + (i + 1) * spf])) for i in range(frames)]
    averages = np.zeros(spf)
    for i in range(spf):
        averages[i] = sum(n_frames[j][i] for j in range(frames)) / frames
    print(len(averages))
    show_img(averages)


def phase_rotation_example_with_noise(noise):
    data = add_noise("data/scene3-640x480-60-400M-64M-40M.dat", "data/noise-640x480-60-400M-64M-40M.dat", noise)
    data = unrotate(data)
    data = resampling(data, f_s, f_r)
    frames = 59
    n_frames = [abs(np.array(data[start + i * spf:start + (i + 1) * spf])) for i in range(frames)]
    averages = np.zeros(spf)
    for i in range(spf):
        averages[i] = sum(n_frames[j][i] for j in range(frames)) / frames
    show_img(averages)


def phase_rotation_example(filename):
    data = load_data(filename)
    data = unrotate(data)
    data = resampling(data, f_s, f_r)
    frames = 59
    n_frames = [abs(np.array(data[start + i * spf:start + (i + 1) * spf])) for i in range(frames)]
    averages = np.zeros(spf)
    for i in range(spf):
        averages[i] = sum(n_frames[j][i] for j in range(frames)) / frames
    show_img(averages)


for i in range(225, 250, 25):
    mhz = str(i)
    start = starts[mhz]
    f_u = f_u_values[mhz]
    hsv_img_with_unrotation_example("data/scene3-640x480-60-" + mhz + "M-64M-40M.dat")
    phase_rotation_example("data/scene3-640x480-60-" + mhz + "M-64M-40M.dat")
# demodulation_example_with_noise(3.5)
# phase_rotation_example_with_noise(3.5)
# hsv_img_with_unrotation_example()
