import numpy as np
import subprocess
import glob


def str_to_float(string):
    try:
        return float(string)
    except ValueError:
        num, denom = string.split('/')
        return float(num) / float(denom)


def str_to_nparray_l(string):
    num_expected = int(string.split()[0])
    tmp_array = np.zeros([num_expected])
    for i in range(num_expected):
        tmp_array[i] = str_to_float(string.split()[i+1])
    return tmp_array


def calc_distortion(fy_orig, x_crop_scale=1.0):
    fy = fy_orig / pow(2, 14) + 1
    n = fy_orig.shape[0]

    x_orig = (np.array(range(n)) + 0.5) / (n-1)
    xscale = np.sqrt(2 * 2 + 3 * 3) / 2  # half height of image is scale 1
    x = x_orig * xscale * x_crop_scale

    y_scale = 1  # initial guess
    for iteration in range(5):
        fy_scaled = fy * y_scale
        x_scaled = x * y_scale
        pos = 0
        for pos in range(x_scaled.shape[0]):
            if x_scaled[pos] >= 1:
                break
        fac_left = (x_scaled[pos] - 1) / (x_scaled[pos] - x_scaled[pos - 1])
        fac_right = 1 - fac_left
        dev = fy_scaled[pos] * fac_right + fy_scaled[pos - 1] * fac_left
        y_scale = y_scale / dev
        # print('dev: ', dev, ', new y-scale: ', y_scale)

    x = x * y_scale
    y = x * fy * y_scale

    sx2 = np.sum(np.power(x, 2))
    sx3 = np.sum(np.power(x, 3))
    sx4 = np.sum(np.power(x, 4))
    sx5 = np.sum(np.power(x, 5))
    sx6 = np.sum(np.power(x, 6))
    sx7 = np.sum(np.power(x, 7))
    sx8 = np.sum(np.power(x, 8))
    syx = np.sum(np.multiply(y, x))
    syx2 = np.sum(np.multiply(y, np.power(x, 2)))
    syx3 = np.sum(np.multiply(y, np.power(x, 3)))
    syx4 = np.sum(np.multiply(y, np.power(x, 4)))

    vxy = np.array([syx4 - syx - sx5 + sx2,
                    syx3 - syx - sx4 + sx2,
                    syx2 - syx - sx3 + sx2]).transpose()

    ax = np.array([(sx8 - sx5 - sx5 + sx2, sx7 - sx5 - sx4 + sx2, sx6 - sx5 - sx3 + sx2),
                   (sx7 - sx5 - sx4 + sx2, sx6 - sx4 - sx4 + sx2, sx5 - sx4 - sx3 + sx2),
                   (sx6 - sx5 - sx3 + sx2, sx5 - sx3 - sx4 + sx2, sx4 - sx3 - sx3 + sx2)])

    coeffs = np.linalg.solve(ax, vxy)
    # a = coeffs[0]
    # b = coeffs[1]
    # c = coeffs[2]
    return coeffs


def calc_vignetting(v_orig, x_crop_scale=1.0):
    y = (pow(2, 14) - v_orig) / pow(2, 14)
    n = v_orig.shape[0]

    x_orig = (np.array(range(n)) + 0.5) / (n-1)
    xscale = 1  # half diagonal of image is scale 1
    x = x_orig * xscale * x_crop_scale

    sx2 = np.sum(np.power(x, 2))
    sx4 = np.sum(np.power(x, 4))
    sx6 = np.sum(np.power(x, 6))
    sx8 = np.sum(np.power(x, 8))
    sx10 = np.sum(np.power(x, 10))
    sx12 = np.sum(np.power(x, 12))
    syx2 = np.sum(np.multiply(y, np.power(x, 2)))
    syx4 = np.sum(np.multiply(y, np.power(x, 4)))
    syx6 = np.sum(np.multiply(y, np.power(x, 6)))

    vxy = np.array([syx2 - sx2,
                    syx4 - sx4,
                    syx6 - sx6]).transpose()

    ax = np.array([(sx4, sx6, sx8),
                   (sx6, sx8, sx10),
                   (sx8, sx10, sx12)])

    coeffs = np.linalg.solve(ax, vxy)

    # k1 = coeffs[0]
    # k2 = coeffs[1]
    # k3 = coeffs[2]

    return coeffs


def calc_tca(fy_orig, x_crop_scale=1.0):
    n = fy_orig.shape[0]
    fy = fy_orig / pow(2, 14 + 7) + 1

    x_orig = (np.array(range(n)) + 0.5) / (n-1)
    xscale = np.sqrt(2 * 2 + 3 * 3) / 2  # half height of image is scale 1
    x = x_orig * xscale * x_crop_scale

    x = x
    y = x * fy

    sx2 = np.sum(np.power(x, 2))
    sx3 = np.sum(np.power(x, 3))
    sx4 = np.sum(np.power(x, 4))
    sx5 = np.sum(np.power(x, 5))
    sx6 = np.sum(np.power(x, 6))
    syx1 = np.sum(np.multiply(y, np.power(x, 1)))
    syx2 = np.sum(np.multiply(y, np.power(x, 2)))
    syx3 = np.sum(np.multiply(y, np.power(x, 3)))

    vxy = np.array([syx1, syx2, syx3]).transpose()

    ax = np.array([(sx2, sx3, sx4),
                   (sx3, sx4, sx5),
                   (sx4, sx5, sx6)])

    coeffs = np.linalg.solve(ax, vxy)

    # v = coeffs[0]
    # c = coeffs[1]
    # b = coeffs[2]

    return coeffs


def print_coeffs_distortion(coeffs):
    print('Distortion: ')
    print('  a = ', coeffs[0])
    print('  b = ', coeffs[1])
    print('  c = ', coeffs[2])


def print_coeffs_vignetting(coeffs):
    print('Vignetting: ')
    print('  k1 = ', coeffs[0])
    print('  k2 = ', coeffs[1])
    print('  k3 = ', coeffs[2])


def print_coeffs_tca(coeffs_red, coeffs_blue):
    print('TCA: ')
    print('  vr = ', coeffs_red[0])
    print('  cr = ', coeffs_red[1])
    print('  br = ', coeffs_red[2])
    print('  vr = ', coeffs_blue[0])
    print('  cr = ', coeffs_blue[1])
    print('  br = ', coeffs_blue[2])


class lens_info:
    def __init__(self):
        self.manufacturer = ''
        self.name = ''
        self.camera_model = ''
        self.focal_length = 0
        self.fnumber = 0
        self.focus_distance = 0
        self.mount = ''
        self.crop = 1.0
        self.x_crop_scale = 1.0
        self.corr_distortion_raw = np.array([0])
        self.corr_tca_r_raw = np.array([0])
        self.corr_tca_b_raw = np.array([0])
        self.corr_vignetting_raw = np.array([0])
        self.corr_distortion = np.array([0])
        self.corr_tca_r = np.array([0])
        self.corr_tca_b = np.array([0])
        self.corr_vignetting = np.array([0])

    def load_from_file(self, filename):
        exif_focal_length = subprocess.check_output(
            ["exiv2", "-K", "Exif.Photo.FocalLength", "-P", "v", filename])
        self.focal_length = str_to_float(str(exif_focal_length)[2:-3])

        exif_fnumber = subprocess.check_output(
            ["exiv2", "-K", "Exif.Photo.FNumber", "-P", "v", filename])
        self.fnumber = str_to_float(str(exif_fnumber)[2:-3])

        exif_name = subprocess.check_output(
            ["exiv2", "-K", "Exif.Photo.LensModel", "-P", "v", filename])
        self.name = str(exif_name)[2:-3]

        exif_camera_model = subprocess.check_output(
            ["exiv2", "-K", "Exif.Image.Model", "-P", "v", filename])
        self.camera_model = str(exif_camera_model)[2:-3]

        exif_focus_distance = subprocess.check_output(
            ["exiv2", "-K", "Exif.Sony2Fp.FocusPosition2", "-P", "v", filename])
        tmp_distance = str_to_float(str(exif_focus_distance)[2:-3])
        exif_focal_length_35mm = subprocess.check_output(
            ["exiv2", "-K", "Exif.Photo.FocalLengthIn35mmFilm", "-P", "v", filename])
        focal_length_35mm = str_to_float(str(exif_focal_length_35mm)[2:-3])
        self.focus_distance = (pow(2, tmp_distance / 16 - 5) + 1) * focal_length_35mm / 1000  # taken from darktable
        # should be checked. Same distance as darktable shows, but these seem so be wrong (at least on zoom lenses)

        exif_focal_length = subprocess.check_output(
            ["exiv2", "-K", "Exif.Photo.FocalLength", "-P", "v", filename])
        focal_length = str_to_float(str(exif_focal_length)[2:-3])
        self.crop = focal_length_35mm / focal_length

        exif_corr_distortion = subprocess.check_output(
            ["exiv2", "-K", "Exif.SubImage1.DistortionCorrParams", "-P", "v", filename])
        self.corr_distortion_raw = str_to_nparray_l(str(exif_corr_distortion)[2:-3])

        exif_corr_vignetting = subprocess.check_output(
            ["exiv2", "-K", "Exif.SubImage1.VignettingCorrParams", "-P", "v", filename])
        self.corr_vignetting_raw = str_to_nparray_l(str(exif_corr_vignetting)[2:-3])

        exif_corr_tca = subprocess.check_output(
            ["exiv2", "-K", "Exif.SubImage1.ChromaticAberrationCorrParams", "-P", "v", filename])
        tmp_tca = str_to_nparray_l(str(exif_corr_tca)[2:-3])
        self.corr_tca_r_raw = tmp_tca[0:(tmp_tca.shape[0] // 2)]
        self.corr_tca_b_raw = tmp_tca[(tmp_tca.shape[0] // 2):]

        # I found that the 11 coefficients in APS-C mode are equal to the first 11 coefficients of the 16 coefficients
        # in full-frame mode.
        # The distance between (full-frame) support points is sqrt(12mm^2 + 18mm^2)/15 = d_ff/15 with d_ff being half
        # the diagonal of the full frame sensor. (See calculation in calc_* functions for details).
        # This is a physical quantity and can not change in APS-C mode (if the coefficients are placed at the same
        # physical position on the sensor - what I assume since they are equal). For vignetting half the diagonal of
        # the image is r=1. Thus, the spacing between the support points is dr_ff = 1/15. For tca and distortion r=1 is
        # half the image height and an additional scaling factor has to be multiplied (just as a reminder - this factor
        # will be the same in APS-C mode).
        # When it comes to APS-C mode, the physical size of the (half) diagonal is d_crop = d_ff/crop.
        # In the crop mode we only have 11 coefficients. The 10 equal distant spacings between them are in the real
        # physical world equal (d_ff/15). In relation to half the crop image diagonal d_crop we get
        # d_ff/15 = d_crop*crop/15 = d_crop/10 * ((10*crop)/15).
        # Thus the spacing dr_crop is not exactly 1/10 (in relative image half diagonal coordinates) but 1/10 multiplied
        # by x_crop_scale = ((10*crop)/15), thus, dr_crop = 1/10 * x_crop_scale.
        # In case of crop = 1.5 the factor x_crop_scale would be 1. However, Sony APS-C mode has a crop factor of
        # about 1.524 which is addressed by x_crop_scale.
        #
        # One short note: Using my Sony A7 III (3.00) in APS-C mode I found that Sony is doing the opposite. Instead of
        # multiplying with x_crop_scale Sony is dividing by this number. As a result images of exactly the same
        # setting taken in APS-C and full frame mode with in-camera distortion correction can not be aligned.
        # When multiplying with x_crop_scale, manually corrected images from both modes can be perfectly aligned.
        #
        # One last note: The difference is not very large (few pixels in the radius shift) and might be neglected

        # check if we are in crop mode (only 11 coefficients)
        x_crop_scale = 1.0
        if self.corr_vignetting_raw.shape[0] == 11:
            x_crop_scale = 1.524/1.5

        self.corr_distortion = calc_distortion(self.corr_distortion_raw, x_crop_scale)

        self.corr_tca_r = calc_tca(self.corr_tca_r_raw, x_crop_scale)

        self.corr_tca_b = calc_tca(self.corr_tca_b_raw, x_crop_scale)

        self.corr_vignetting = calc_vignetting(self.corr_vignetting_raw, x_crop_scale)

    def set_manufacturer(self, manufacturer):
        self.manufacturer = manufacturer

    def set_mount(self, mount):
        self.mount = mount


def indent(depth):
    return ' ' * (4*depth)


def to_lf_distortion(lens_mod: lens_info):
    out = indent(3)
    out += '<distortion model="ptlens" '
    out += 'focal="{:.0f}" '.format(lens_mod.focal_length)
    out += 'a="{:.7f}" '.format(lens_mod.corr_distortion[0])
    out += 'b="{:.7f}" '.format(lens_mod.corr_distortion[1])
    out += 'c="{:.7f}" '.format(lens_mod.corr_distortion[2])
    out += '/>\n'
    return out


def to_lf_tca(lens_mod: lens_info):
    out = indent(3)
    out += '<tca model="poly3" '
    out += 'focal="{:.0f}" '.format(lens_mod.focal_length)
    out += 'vr="{:.7f}" '.format(lens_mod.corr_tca_r[0])
    out += 'cr="{:.7f}" '.format(lens_mod.corr_tca_r[1])
    out += 'br="{:.7f}" '.format(lens_mod.corr_tca_r[2])
    out += 'vb="{:.7f}" '.format(lens_mod.corr_tca_b[0])
    out += 'cb="{:.7f}" '.format(lens_mod.corr_tca_b[1])
    out += 'bb="{:.7f}" '.format(lens_mod.corr_tca_b[2])
    out += '/>\n'
    return out


def to_lf_vignetting(lens_mod: lens_info):
    out = indent(3)
    out += '<vignetting model="pa" '
    out += 'focal="{:.0f}" '.format(lens_mod.focal_length)
    out += 'aperture="{:.1f}" '.format(lens_mod.fnumber)
    out += 'distance="{:.2f}" '.format(lens_mod.focus_distance)
    out += 'k1="{:.7f}" '.format(lens_mod.corr_vignetting[0])
    out += 'k2="{:.7f}" '.format(lens_mod.corr_vignetting[1])
    out += 'k3="{:.7f}" '.format(lens_mod.corr_vignetting[2])
    out += '/>\n'
    return out


def to_lf_head(lens_mod: lens_info):
    out = '<lensdatabase>\n\n'
    out += indent(1) + '<lens>\n'
    out += indent(2) + '<maker>' + lens_mod.manufacturer + '</maker>\n'
    out += indent(2) + '<model>' + lens_mod.name + '</model>\n'
    out += indent(2) + '<mount>' + lens_mod.mount + '</mount>\n'
    out += indent(2) + '<cropfactor>' + str(lens_mod.crop) + '</cropfactor>\n'
    out += indent(2) + '<calibration>\n'
    out += indent(3) + '<!-- Taken with ' + lens_mod.camera_model + ', coeffients fitted from exif data by lf_fitexif_se -->\n'
    return out


def to_lf_foot():
    out = indent(2) + '</calibration>\n'
    out += indent(1) + '</lens>\n\n'
    out += '</lensdatabase>\n'
    return out


# MAIN

lens_mod = lens_info()
lens_mod.set_manufacturer('FILL IN')
lens_mod.set_mount('Sony E')

is_data = False

# Distortion:

file_list = glob.glob('distortion/*.[aA][rR][wW]')
file_list.sort()
out_dist = ''

for i in range(len(file_list)):
    lens_mod.load_from_file(file_list[i])
    out_dist += to_lf_distortion(lens_mod)
    is_data = True

# TCA:

file_list = glob.glob('tca/*.[aA][rR][wW]')
file_list.sort()
out_tca = ''

for i in range(len(file_list)):
    lens_mod.load_from_file(file_list[i])
    out_tca += to_lf_tca(lens_mod)
    is_data = True

# Vignetting:

file_list = glob.glob('vignetting/*.[aA][rR][wW]')
file_list.sort()
out_vignetting = ''

for i in range(len(file_list)):
    lens_mod.load_from_file(file_list[i])
    out_vignetting += to_lf_vignetting(lens_mod)
    is_data = True

if is_data:
    output = to_lf_head(lens_mod) + out_dist + out_tca + out_vignetting + to_lf_foot()
    print(output)
    file_results = open('lensfun_exif.xml', 'w')
    file_results.write(output)
    file_results.close()



