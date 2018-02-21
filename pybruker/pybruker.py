import numpy as np
import nibabel as nib
import os
import re
import datetime
import dateutil


class BrukerRaw:
    """ BrukerRaw class
    This class provides the function to import parameters, data from Bruker Raw data
    """
    def __init__(self, path, summary=False):
        """ Initiate the object
        """
        self.path = path
        try:
            with open(os.path.join(path, 'subject')) as f:
                subject = f.readlines()[:]
                self.subject = self.parsing(subject)
            pattern = r'Parameter List, ParaVision (.*)'
            if re.match(pattern, self.subject['TITLE']):
                self.version = re.sub(pattern, r'\1', self.subject['TITLE'])
            else:
                self.version = 'Paravision 5.1'
            # Check all scan information
            self.check_scans()
            if summary:
                self.summary()
        except:
            self.subject = None
            self.scans = None

    def method_param(self, scan):
        """ Check scan method parameter

        :param scan: Scan ID
        :return: TR, TE, FlipAngle, BandWidth
        """
        method = self.scans[str(scan)]['method']
        acqp = self.scans[str(scan)]['acqp']
        tr = method['PVM_RepetitionTime']
        te = method['PVM_EchoTime']
        fa = acqp['ACQ_flip_angle']
        try:
            bw = method['PVM_EffSWh']/1000.0
        except:
            bw = 0.0
        return tr, te, bw, fa

    def check_scans(self):
        """ Check all scans and save the parameter into self.scans object as dictionary
        collecting information
        - method
        - acqp
        - visu_pars
        """
        self.scans = dict()
        for scan in os.listdir(self.path):
            scan_path = os.path.join(self.path, scan)
            if os.path.isdir(scan_path):
                self.scans[scan] = dict()
                try:
                    with open(os.path.join(scan_path, 'method')) as f:
                        method = f.readlines()[:]
                    with open(os.path.join(scan_path, 'acqp')) as f:
                        acqp = f.readlines()[:]
                    self.scans[scan]['method'] = self.parsing(method)
                    self.scans[scan]['acqp'] = self.parsing(acqp)
                    self.scans[scan]['reco'] = dict()
                    self.scans[scan]['recoparam'] = dict()
                    for reco in os.listdir(os.path.join(scan_path, 'pdata')):
                        reco_path = os.path.join(scan_path, 'pdata', reco)
                        if os.path.isdir(reco_path):
                            with open(os.path.join(reco_path, 'visu_pars')) as f:
                                pars = f.readlines()[:]
                            with open(os.path.join(reco_path, 'reco')) as f:
                                recoparam = f.readlines()[:]
                            self.scans[scan]['reco'][reco] = self.parsing(pars)
                            self.scans[scan]['recoparam'][reco] = self.parsing(recoparam)
                except:
                    del self.scans[scan]

    def load_data(self, scan, reco):
        """ Load data into numpy array
        """
        code = self.get_code(scan, reco)
        data_path = os.path.join(self.path, str(scan), 'pdata', str(reco), '2dseq')
        data = np.fromfile(data_path, dtype=np.dtype(code))
        return data

    def get_code(self, scan, reco):
        """ Check datatype and generate the code
        This function is immature, need to do more test in case Paravision 6 uses different data type
        (maybe for complex number?)
        """
        pars = self.scans[str(scan)]['reco'][str(reco)]
        bite_order = pars['VisuCoreByteOrder']
        if bite_order == 'littleEndian':
            code = '<'
        elif bite_order == 'bigEndian':
            code = '>'
        else:
            code = '='
        word_type = pars['VisuCoreWordType'].split('_')[1:]
        if word_type[2] == 'INT':
            if word_type[1] == 'SGN':
                w_type = 'i'
            else:
                w_type = 'u'
        else:
            w_type = 'f'
        code = '{}{}'.format(code, w_type)
        bit_pattern = r'(\d+)\w+'
        code = '{}{}'.format(code, int(re.sub(bit_pattern, r'\1', word_type[0])) / 8)
        return code

    def get_size(self, scan, reco):
        """ Calculate the information of the metrix size
        dimmension of the acqusition protocol, shape, number of slice packages,
        and index of slice_pack on shape will be returned.
        """
        pars = self.scans[str(scan)]['reco'][str(reco)]
        method = self.scans[str(scan)]['method']
        matrix = pars['VisuCoreSize']
        if not isinstance(matrix, list):
            matrix = [matrix]
        shape = matrix[:]
        dim = pars['VisuCoreDim']

        if self.check_version(scan, reco) == 1: # Check number of slice packages
            if isinstance(method['PVM_SPackArrSliceOrient'], list):
                slice_packs = len(method['PVM_SPackArrSliceOrient'])
            else:
                slice_packs = 1
        else:
            if 'VisuCoreSlicePacksDef' in pars.keys():
                slice_packs = pars['VisuCoreSlicePacksDef'][1]
            else:
                slice_packs = 1

        if 'VisuCoreSlicePacksSlices' in pars.keys(): # Calculate number of slices
            slices_param = pars['VisuCoreSlicePacksSlices']
            if len(pars['VisuCoreSlicePacksSlices']) > 1:
                if isinstance(slices_param[0], int):
                    slices = slices_param[1]
                else:
                    slices = slices_param[-1][-1]
            else:
                slices = len(pars['VisuCoreOrientation'])
        else:
            if 'VisuCoreOrientation' in pars.keys():
                slices = len(pars['VisuCoreOrientation']) / slice_packs
            else:
                slices = 1

        if slices == 1: # Complete 3D matrix shape
            if dim < 3:
                shape.append(slices)
        else:
            shape.append(slices)

        # Add rest of frame as 4th dimension
        frame = pars['VisuCoreFrameCount'] / slices / slice_packs
        if slice_packs > 1:
            shape.append(slice_packs)
            idx = len(shape) - 1
        else:
            idx = None
        if frame > 1:
            shape.append(frame)
        return dim, shape, slice_packs, idx

    def get_img(self, scan, reco):
        """ Reshape loaded data into image structure

        :param scan: Scan ID
        :param reco: Reco ID
        :return: Shaped image
        """
        data = self.load_data(scan, reco)
        dim, size, slice_packs, idx = self.get_size(scan, reco)
        img = data.reshape(size[::-1]).T
        _, inverted = self.get_center(scan, reco)
        if inverted:
            img = img[:, :, ::-1, ...]
        return img

    def save_as(self, scan, reco, filename, ext='.nii.gz'):
        """ Save image data into NifTi format

        :param scan: Scan ID
        :param reco: Reco ID
        :param filename: Output filename without extension
        :param ext: extension (default: .nii.gz)
        :return: Boolean (if 1, image is not saved)
        """
        dim, shape, slice_packs, idx = self.get_size(scan, reco)
        if dim < 2:
            pass
        else:
            img = self.get_img(scan, reco)
            affine = self.calc_affine(scan, reco)
            if slice_packs == 1:
                nii = nib.Nifti1Image(img, affine)
                nii = self.set_default_header(nii, scan)
                nii.to_filename('{}{}'.format(filename, ext))
            else:
                img = np.swapaxes(img, idx, -1)
                slice_orient = self.scans[str(scan)]['method']['PVM_SPackArrSliceOrient']
                for i in range(slice_packs):
                    nii = nib.Nifti1Image(img[..., i], affine[i])
                    nii = self.set_default_header(nii, scan)
                    nii.to_filename('{}_{}{}'.format(filename, slice_orient[i], ext))
            return 0
        return 1

    def get_resol(self, scan, reco):
        """ Calculate image resolution

        :param scan: Scan ID
        :param reco: Reco ID
        :return: Image resolution
        """
        dim, shape, slice_packs, idx = self.get_size(scan, reco)
        pars = self.scans[str(scan)]['reco'][str(reco)]
        FOV = pars['VisuCoreExtent'][:]
        orient = self.scans[str(scan)]['method']['PVM_SPackArrSliceOrient']
        try: # Use slice distance as final slice tickness
            dist = pars['VisuCoreSlicePacksSliceDist']
        except:
            dist = pars['VisuCoreFrameThickness']
        if isinstance(dist, list):
            dist = dist[0]

        if dim == 2:
            FOV.append(shape[2] * dist)
        try: # calculate resolution and correct axes
            resol = np.asarray(map(float, FOV)) / np.asarray(shape[:3])
            if slice_packs == 1:
                tf = self.check_transform(scan, reco)
                return list(map(abs, np.dot(resol, tf)))
            else:
                resols = []
                tf = self.check_transform(scan, reco)
                for i in range(slice_packs):
                    if orient[i] == 'sagittal':
                        resols.append(map(abs, np.dot(resol, tf[i])))
                    else:
                        resols.append(map(abs, np.dot(resol, tf[i].T)))
                return resols
        except:
            return None

    def get_center(self, scan, reco):
        """ Check center coordinate

        :param scan: Scan ID
        :param reco: Reco ID
        :return: Center coordinate, slice orientation inversion (Boolean)
        """
        dim, shape, slice_packs, idx = self.get_size(scan, reco)
        pars = self.scans[str(scan)]['reco'][str(reco)]
        if slice_packs == 1:
            first = pars['VisuCorePosition'][0] # Check if slice orientation is inverted
            last = pars['VisuCorePosition'][-1]
            if first.sum() - last.sum() < 0:
                return first, False
            else:
                return last, True
        else:
            return pars['VisuCorePosition'], False

    def calc_affine(self, scan, reco, quadruped=True):
        """ Calculate affine transformation matrix

        :param scan: Scan ID
        :param reco: Reco ID
        :param quadruped: True if animal type is quadruped for correct position
        :return: Affine transformation matrix
        """
        dim, shape, slice_packs, idx = self.get_size(scan, reco)
        pars = self.scans[str(scan)]['reco'][str(reco)]
        resol = self.get_resol(scan, reco)
        tf = self.check_transform(scan, reco)
        center, inverted = self.get_center(scan, reco)
        orient = self.scans[str(scan)]['method']['PVM_SPackArrSliceOrient']

        if self.check_version(scan, reco) == 1:  # for PV5.1 and below
            if quadruped == True:
                correct = np.array([[1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]])
            else:
                correct = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        else:
            if pars['VisuSubjectType'] == 'Quadruped': # for PV6 and above
                correct = np.array([[1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]])
            else:
                correct = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        if slice_packs == 1: # If image has only one slice package, return one affine matrix
            if isinstance(resol, list):
                affine = np.dot(np.diag(resol), tf.T)
                affine = np.append(affine.T, [center], axis=0).T
                affine = np.append(affine, [[0, 0, 0, 1]], axis=0)
                affine = np.dot(affine.T, np.diag([1, -1, 1, 1])).T
                affine = np.dot(affine.T, correct).T
            else:
                affine = None
        else:
            if isinstance(resol, list): # If image has multiple slice packages, return list of affine matrices
                affine = []
                for i in range(slice_packs):
                    temp = np.dot(np.diag(resol[i]), tf[i].T)
                    temp = np.append(temp.T, [center[i]], axis=0).T
                    temp = np.append(temp, [[0, 0, 0, 1]], axis=0)
                    temp = np.dot(temp.T, np.diag([1, -1, 1, 1])).T
                    affine.append(np.dot(temp.T, correct).T)
            else:
                affine = None
        return affine

    def check_transform(self, scan, reco):
        """ Return transform matrix saved on VisuPars

        :param scan: Scan ID
        :param reco: Reco ID
        :return: Transform matrix
        """
        tfs = self.scans[str(scan)]['reco'][str(reco)]['VisuCoreOrientation']
        result = tfs == tfs[0, :]
        if result.all():
            return tfs[0, :].reshape([3, 3])
        else:
            return tfs.reshape([tfs.shape[0], 3, 3])

    @property
    def scanned(self):
        """ Return acquired scan numbers """
        try:
            return sorted(map(int, self.scans.keys()))
        except:
            return []

    def check_method(self, scan):
        """ Check acquisition method """
        return self.scans[str(scan)]['method']['Method']

    def parsing(self, profiles):
        """ Parsing parameter files into Dictionary
        Some method paramters include '##$PVM_FlowSatGeoCub' are not compatible.
        There is no reason to fix it yet, no plan to fix this issue

        :param profiles: list of strings
        :return: Dictionary
        """
        p_mparam = r'^\#\#([^$]*)\=(.*)'
        p_profile = r'^\#\#\$(.*)\=.*'
        p_sprofile = r'^\#\#\$(.*)\=([^(].*[^)])'
        p_vprofile = r'^\#\#\$(.*)\=\((.*)\)'
        p_lprofile = r'^\#\#\$(.*)\=\((.*)[^)]$'

        p_vis = r'^\$\$.*'
        p_string = r'^\<(.*)\>$'
        output_obj = dict()

        for i, line in enumerate(profiles):
            if re.search(p_sprofile, line):
                # Parameter has only one value
                key = re.sub(p_sprofile, r'\1', line).strip()
                value = re.sub(p_sprofile, r'\2', line).strip()
                value = self.check_dt(value)
                output_obj[key] = value
            elif re.search(p_vprofile, line):
                # Parameter has multiple values
                key = re.sub(p_vprofile, r'\1', line).strip()
                n_value = re.sub(p_vprofile, r'\2', line).strip()
                try:  # If all value are integer numbers
                    n_value = map(int, map(str.strip, n_value.split(',')))
                except:  # If not, treat as string
                    n_value = map(str, map(str.strip, n_value.split(',')))
                if len(n_value) == 1:  # If parameter has only one value
                    n_value = n_value[0]
                values = list()
                # Check next line if the data are all need to be integrated with current key
                for next_line in profiles[i + 1:]:
                    if re.search(p_profile, next_line) or re.search(p_vis, next_line) or re.search(p_mparam, next_line):
                        break
                    else:
                        values.append(next_line.strip())
                if len(values):
                    values = ' '.join(values)  # murge value together
                    if isinstance(n_value, list):
                        try:
                            values = np.array(self.check_array(n_value, values)).reshape(n_value)
                        except:
                            values = self.check_dt(values)
                        output_obj[key] = values
                    else:
                        if re.match(p_string, values):
                            output_obj[key] = re.sub(p_string, r'\1', values)
                        else:
                            if n_value == 1:
                                values = self.check_dt(values)
                            else:
                                try:
                                    values = self.check_array(n_value, values)
                                except:
                                    print('{}({})={}'.format(key, n_value, values))
                            output_obj[key] = values
                else:
                    output_obj[key] = n_value
            elif re.search(p_lprofile, line):
                line = [line.rstrip()]
                for next_line in profiles[i + 1:]:
                    if re.search(p_profile, next_line) or re.search(p_vis, next_line):
                        break
                    else:
                        line.append(next_line.rstrip())
                line = ' '.join(line)
                key = re.sub(p_vprofile, r'\1', line).strip()
                value = re.sub(p_vprofile, r'\2', line).strip().split(',')
                for i, v in enumerate(value):
                    value[i] = self.check_dt(v)
                output_obj[key] = value
            elif re.search(p_mparam, line):
                key = re.sub(p_mparam, r'\1', line).strip()
                value = re.sub(p_mparam, r'\2', line).strip()
                output_obj[key] = value
            else:
                pass
        return output_obj

    def check_dt(self, value):
        """ Check datatype of the given PV parameter value

        :param value: String form of parameter value
        :return: Value object with corrected data type
        """
        p_int = r'^-?[0-9]+$'
        p_float = r'^-?(\d+\.?)?\d+([eE][-+]?\d+)?$'
        p_string = r'^[^<(].*[^>)]$'
        p_brk_string = r'^\<([^<>]*)\>$'
        p_list_brk_string = r'^\<.*\>.*\>$'
        p_list = r'^\((.*)\)$'
        value = value.strip(' ')
        if re.match(p_float, value):
            if re.match(p_int, value):
                value = int(value)
            else:
                value = float(value)
        else:
            try:
                value = int(value)
            except:
                if re.match(p_brk_string, value):
                    value = re.sub(p_brk_string, r'\1', value).strip(" ")
                elif re.match(p_list, value):
                    value = re.sub(p_list, r'\1', value).strip(" ")
                    value = value.split(',')
                    for i, v in enumerate(value):
                        v = v.strip()
                        if re.match(p_brk_string, v):
                            value[i] = re.sub(p_brk_string, r'\1', v)
                        else:
                            try:
                                value[i] = int(v)
                            except:
                                value[i] = str(v).strip(' ')
                else:
                    if re.match(p_string, value):
                        pass
                    else:
                        if re.match(p_list_brk_string, value):
                            value = value.split()
                            for i, v in enumerate(value):
                                value[i] = re.sub(p_brk_string, r'\1', v).strip(" ")
                        else:
                            pass
        return value

    def scan_datetime(self, item='all'):
        """ Check scanned date and time

        :param item: Multiple options
                - all : return date, start_time, end_time
                - date : return only date
                - time or duration : return start_time and end_time
        """
        last_scan = str(self.scanned[-1])
        if self.check_version(last_scan, 1) == 1:
            pattern = r'(\d{2}:\d{2}:\d{2})\s(\d{2}\s\w+\s\d{4})'
            start_time = self.subject['SUBJECT_date']
            acq_time = self.scans[last_scan]['reco']['1']['VisuAcqScanTime'] / 1000
            last_scan_time = self.scans[last_scan]['reco']['1']['VisuAcqDate']
            date = dateutil.parser.parse(re.sub(pattern, r'\2', start_time)).date()
            start_time = datetime.time(*map(int, re.sub(pattern, r'\1', start_time).split(':')))
            last_scan_time = datetime.time(*map(int, re.sub(pattern, r'\1', last_scan_time).split(':')))
            time_delta = datetime.timedelta(0, acq_time)
            end_time = (datetime.datetime.combine(date, last_scan_time) + time_delta).time()
        else:
            pattern = r'(\d{4}-\d{2}-\d{2})[T](\d{2}:\d{2}:\d{2})'
            start_time = self.subject['SUBJECT_date'].split(',')[0]
            end_time = self.scans[last_scan]['reco']['1']['VisuCreationDate'].split(',')[0]
            date = datetime.date(*map(int, re.sub(pattern, r'\1', start_time).split('-')))
            start_time = datetime.time(*map(int, re.sub(pattern, r'\2', start_time).split(':')))
            end_time = datetime.time(*map(int, re.sub(pattern, r'\2', end_time).split(':')))
        if item == 'all':
            return date, start_time, end_time
        elif item == 'date':
            return date
        elif item == 'time' or item == 'duration':
            return start_time, end_time


    @property
    def study_id(self):
        """ Subject ID """
        return self.subject['SUBJECT_id']

    @property
    def session_id(self):
        """ Session ID """
        return self.subject['SUBJECT_study_name']

    @property
    def user_name(self):
        """ Name of Researcher """
        return self.subject['SUBJECT_name_string']

    def check_version(self, scan, reco):
        """ Check version of ParaVision

        :param scan: Scan ID
        :param reco: Reco ID
        :return: 1 - PV5.1 or below, 3- PV6 or above
        """
        return self.scans[str(scan)]['reco'][str(reco)]['VisuVersion']

    def check_array(self, n_value, values):
        """ Check if the parameter is array shape #TODO: need to improve but no reason to do it

        :param n_value: Number of values shown on header file
        :param values: The string values which can be split into array
        :return:
        """
        p_groups = r'\(([^)]*)\)'
        if re.match(p_groups, values):
            values = re.findall(p_groups, values)
            values = [map(self.check_dt, value.split(', ')) for value in values]
        else:
            values = map(self.check_dt, values.split())
        return values

    def multiply_shape(self, shape):
        """  Multiply all input

        :param shape: list of int
        :return: results of multiply
        """
        output = 1
        for i in shape:
            output *= i
        return output

    def calc_tempresol(self, scan):
        """ Calculate temporal resolution

        :param scan: Scan ID
        :return: Temporal resolution
        """
        method = self.scans[str(scan)]['method']
        tr = method['PVM_RepetitionTime']
        num_avr = method['PVM_NAverages']
        try:
            num_seg = method['NSegments']
            return tr * num_seg * num_avr
        except:
            return tr * num_avr

    def subject_info(self):
        """ Return subject information """
        type = self.subject['SUBJECT_type']
        gender = self.subject['SUBJECT_sex']
        weight = self.subject['SUBJECT_weight']
        position = self.subject['SUBJECT_position']
        entry = self.subject['SUBJECT_entry']
        dob = self.subject['SUBJECT_dbirth']
        return [type, gender, dob], [position, entry], weight

    def set_default_header(self, nii, scan):
        """ Update NifTi Header information

        :param nii: Nibabel NifTi Object
        :param scan: Scan ID
        :return: Nibabel NifTi Object with updated Header
        """
        nii.header.default_x_flip = False
        method = self.scans[str(scan)]['method']
        acqp = self.scans[str(scan)]['acqp']
        tr = self.calc_tempresol(scan)
        if re.search('EPI', method['Method'], re.IGNORECASE) and not re.search('DTI', method['Method'], re.IGNORECASE):
            nii.header.set_xyzt_units('mm', 'sec')
            nii.header['pixdim'][4] = float(tr) / 1000
            nii.header.set_dim_info(slice=2)
            nii.header['slice_duration'] = float(tr) / (1000 * acqp['NSLICES'])
            if method['PVM_ObjOrderScheme'] == 'User_defined_slice_scheme':
                nii.header['slice_code'] = 0
            elif method['PVM_ObjOrderScheme'] == 'Sequential':
                nii.header['slice_code'] = 1
            elif method['PVM_ObjOrderScheme'] == 'Reverse_sequential':
                nii.header['slice_code'] = 2
            elif method['PVM_ObjOrderScheme'] == 'Interlaced':
                nii.header['slice_code'] = 3
            elif method['PVM_ObjOrderScheme'] == 'Reverse_interlacesd':
                nii.header['slice_code'] = 4
            elif method['PVM_ObjOrderScheme'] == 'Angiopraphy':
                nii.header['slice_code'] = 0
            nii.header['slice_start'] = min(acqp['ACQ_obj_order'])
            nii.header['slice_end'] = max(acqp['ACQ_obj_order'])
        else:
            nii.header.set_xyzt_units('mm', 'unknown')
            nii.header['qform_code'] = 1
            nii.header['sform_code'] = 0
        return nii

    def summary(self):
        """ Print out brief summary of the raw data
        """
        title = '\nParavision {}'.format(self.version)
        print(title)
        print('-' * len(title))
        print('\nYou are looking the summary of [{}]\n'.format(self.path))
        if len(self.scanned):
            date, start_time, end_time = self.scan_datetime()
            print('UserAccount:\t{}'.format(self.subject['OWNER']))
            print('Researcher:\t{}'.format(self.user_name))
            print('Date:\t\t{}'.format(date))
            print('Scan duration:\t{} - {}'.format(start_time, end_time))
            print('Subject ID:\t{}'.format(self.subject['SUBJECT_id']))
            print('Session ID:\t{}'.format(self.subject['SUBJECT_study_name']))
            print('Study ID:\t{}'.format(self.subject['SUBJECT_study_nr']))
            sub_info, pos, weight = self.subject_info()
            print('Subject Type:\t{}'.format(sub_info[0]))
            print('Gender:\t\t{}'.format(sub_info[1]))
            print('Date of Birth:\t{}'.format(sub_info[2]))
            print('weight:\t\t{} kg'.format(weight))
            print('Position:\t{}\t\tEntry:\t{}'.format(pos[0], pos[1]))
            print('\nScanID\tSequence::Protocol::[Parameters]')
            for scan in self.scanned:
                acqp = self.scans[str(scan)]['acqp']
                # method = self.scans[str(scan)]['method']
                scan_method = acqp['ACQ_method']
                if ':' in scan_method or ' ' in scan_method:
                    scan_method = acqp['ACQ_method'].split(':')[-1]
                params = self.method_param(scan)
                params = "[ TR: {0}ms, TE: {1:.2f}ms, BW: {2:.2f}kHz, FlipAngle: {3} ]".format(*params)
                print('{}:\t{}::{}::{}'.format(str(scan).zfill(3),
                                               scan_method,
                                               acqp['ACQ_scan_name'],
                                               params))
                for reco in self.scans[str(scan)]['reco'].keys():
                    dim, size, slice_packs, idx = self.get_size(scan, reco)
                    recoparam = self.scans[str(scan)]['recoparam'][str(reco)]
                    mmode = recoparam['RECO_map_mode']
                    range = "{} ~ {}".format(*recoparam['RECO_map_range'])
                    if dim > 1:
                        if idx:
                            del size[idx]
                            resol = ','.join(['{}'.format(round(i, 2)) for i in self.get_resol(scan, reco)[0]])
                            size = 'x'.join(map(str, size))
                            size = '{}, resol(mm): {}, slice_packs: {}'.format(size, resol, slice_packs)
                        else:
                            resol = ','.join(['{}'.format(round(i, 2)) for i in self.get_resol(scan, reco)])
                            size = 'x'.join(map(str, size))
                            size = '{}, resol(mm): {}'.format(size, resol)
                        print('\t[{}] dim: {}D, size: {}, \n'
                              '\t     mapmode: {}, range: {}'.format(str(reco).zfill(2),
                                                                                         dim, size,
                                                                                         mmode, range))
                    else:
                        print('\t[{}] dim: {}D, size: {}, \n'
                              '\t     mapmode: {}, range: {}'.format(str(reco).zfill(2),
                                                                                         dim, size,
                                                                                         mmode, range))
        else:
            print('Empty study...')