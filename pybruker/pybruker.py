import numpy as np
import nibabel as nib
import os
import re
import datetime
import logging
import sys


'''
##########################################################################################################
#  BrukerHandler: parsing, get value - No changing
##########################################################################################################
'''


class BrukerHandler:
    """ BrukerHandler class
    This class provides parsing, getting data from Bruker raw file.
    """

    # ==============================================================================
    #  Initiator
    # ==============================================================================
    def __init__(self, log_handler='', raise_exp=False):
        """ Initiate the object
        """
        self.raise_exp = raise_exp

        if log_handler:
            self.logger = logging.getLogger(log_handler)
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter('%(module)s.%(funcName)s(line:%(lineno)d) %(message)s')
            handler.setFormatter(formatter)
            self.logger.handlers = [handler]

    # ==============================================================================
    #  log
    # ==============================================================================
    def log(self, msg):
        """ Initiate the object
        """
        self.logger.debug(msg)

    # ==============================================================================
    #  parsing: parameter files to dictionary[key:value] object
    # ==============================================================================
    def parsing(self, profiles):
        """ Parsing parameter files into Dictionary
        Some method paramters include '##$PVM_FlowSatGeoCub' are not compatible.
        There is no reason to fix it yet, no plan to fix this issue

        :param profiles: list of strings
        :return: Dictionary
        """

        # regular expression
        p_mparam = r'^\#\#([^$]*)\=(.*)'
        p_profile = r'^\#\#\$(.*)\=.*'
        p_sprofile = r'^\#\#\$(.*)\=([^(].*[^)])'
        p_vprofile = r'^\#\#\$(.*)\=\((.*)\)'
        p_lprofile = r'^\#\#\$(.*)\=\((.*)[^)]$'
        p_vis = r'^\$\$.*'
        p_string = r'^\<(.*)\>$'

        # return object
        output_obj = dict()

        for i, line in enumerate(profiles):

            # 1) Parameter has only one value: ex) ##$SUBJECT_version_nr=2
            if re.search(p_sprofile, line):
                key = re.sub(p_sprofile, r'\1', line).strip()
                value = re.sub(p_sprofile, r'\2', line).strip()
                value = self.conversion_datatype(value)
                output_obj[key] = value

            # 2) Parameter has multiple values: ex) ##$PVM_Nucleus1=( 8 ) <1H>
            elif re.search(p_vprofile, line):
                key = re.sub(p_vprofile, r'\1', line).strip()
                n_value = re.sub(p_vprofile, r'\2', line).strip()
                values = list()

                try:
                    # If all value are integer numbers
                    n_value = map(int, map(str.strip, n_value.split(',')))
                except ValueError:
                    # If not, treat as string
                    n_value = map(str, map(str.strip, n_value.split(',')))
                except Exception as e:
                    if self.raise_exp:
                        raise e

                # If parameter has only one value
                if len(n_value) == 1:
                    n_value = n_value[0]

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
                            values = np.array(self.conversion_array(values)).reshape(n_value)
                        except ValueError:
                            values = self.conversion_datatype(values)
                        except Exception as e:
                            if self.raise_exp:
                                raise e
                        output_obj[key] = values
                    else:
                        if re.match(p_string, values):
                            output_obj[key] = re.sub(p_string, r'\1', values)
                        else:
                            if n_value == 1:
                                values = self.conversion_datatype(values)
                            else:
                                try:
                                    values = self.conversion_array(values)
                                except ValueError:
                                    self.log('{}({})={}'.format(key, n_value, values))
                                except Exception as e:
                                    if self.raise_exp:
                                        raise e
                            output_obj[key] = values
                else:
                    output_obj[key] = n_value

            # 3) Parameter has multiple values like tuple: ex) ##$ExcPulse=(1, 2, 3,)
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
                for k, v in enumerate(value):
                    value[k] = self.conversion_datatype(v)
                output_obj[key] = value

            # 4) Parameter has one value: ex) ##OWNER=nmrsu
            elif re.search(p_mparam, line):
                key = re.sub(p_mparam, r'\1', line).strip()
                value = re.sub(p_mparam, r'\2', line).strip()
                output_obj[key] = value
            # 5) the others
            else:
                pass

        return output_obj

    # ==============================================================================
    #  conversion_datatype: Change the string value to the appropriate data type value
    # ==============================================================================
    def conversion_datatype(self, value):
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
            except ValueError:
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
                            except ValueError:
                                value[i] = str(v).strip(' ')
                            except Exception as e2:
                                if self.raise_exp:
                                    raise e2
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
            except Exception as e:
                if self.raise_exp:
                    raise e
        return value

    # ==============================================================================
    #  conversion_array: Change the string value to the appropriate data type of array
    # ==============================================================================
    def conversion_array(self, values):
        """ Check if the parameter is array shape
        :param values: The string values which can be split into array
        :return:
        """

        p_groups = r'\(([^)]*)\)'

        # multiple array which is split first with ' '(space) and second with ()
        # ex) (<A>, 50, 0) (<B>, 30, 0) => [['A', 50, 0], ['B', 30, 0]]
        if re.match(p_groups, values):
            values = re.findall(p_groups, values)
            values = [map(self.conversion_datatype, value.split(', ')) for value in values]

        # single array which can be split with ' '(space):
        # ex) 304 96 52 -> [302, 96, 52]
        else:
            values = map(self.conversion_datatype, values.split())

        return values


'''
##########################################################################################################
#  BrukerRawBase
##########################################################################################################
'''


class BrukerRawBase(BrukerHandler):
    """ BrukerRaw class
    This class provides the function to import parameters, data from Bruker Raw data
    """

    def __init__(self, log_handler='', raise_exp=False):
        """ Initiate the object
        """
        self.log_handler = log_handler
        self.raise_exp = raise_exp

        # Super Class
        BrukerHandler.__init__(self, self.log_handler, self.raise_exp)

    # ==============================================================================
    #  get_raw_object: return dic() of raw file
    # ==============================================================================
    def get_raw_object(self, source_path, file_name):

        # 'subject' file open
        with open(os.path.join(source_path, file_name)) as f:
            file_line = f.readlines()[:]

            # subject dictionary
            dic_raw = self.parsing(file_line)
            # self.log("file_name==" + file_name)
            # self.log(dic_raw)

        return dic_raw

    # ==============================================================================
    #  get_version: return string version
    # ==============================================================================
    def get_version(self, dic_subject):

        # Get Version
        pattern = r'Parameter List, ParaVision (.*)'
        if re.match(pattern, dic_subject['TITLE']):
            str_version = re.sub(pattern, r'\1', dic_subject['TITLE'])
        else:
            str_version = 'Paravision 5.1'
        # self.log("## version == [" + str_version + "]")

        return str_version

    # ==============================================================================
    #  get_scan_list_from_path: return list scan number
    # ==============================================================================
    def get_scan_list_from_path(self, source_path):
        """
        Return acquired scan numbers
        :param source_path:
        :return: list
        """
        scan_list = list()
        try:
            for scan in os.listdir(source_path):
                scan_path = os.path.join(source_path, scan)
                if os.path.isdir(scan_path):
                    scan_list.append(scan)

            return sorted(map(int, scan_list))

        except AttributeError:
            return []
        except ValueError:
            return []
        except Exception as e:
            if self.raise_exp:
                raise e

    # ==============================================================================
    #  get_scan_list: return list scan number
    # ==============================================================================
    def get_scan_list(self, dic_scan):
        """
         Return acquired scan numbers
        :param dic_scan:
        :return: list
        """
        try:
            return sorted(map(int, dic_scan.keys()))
        except AttributeError:
            return []
        except ValueError:
            return []
        except Exception as e:
            if self.raise_exp:
                raise e

    # ==============================================================================
    #  get_dic_acqp: return acqp values
    # ==============================================================================
    def get_dic_acqp(self, scans, scan_num):
        """
        return acqp values
        :param scans: Scans
        :param scan_num: Scan ID
        :return:
        """
        scan_acqp = scans[str(scan_num)]['acqp']

        # return Dictionary
        dic_acqp = dict()

        #-------method
        acq_method = scan_acqp['ACQ_method']
        if ':' in acq_method or ' ' in acq_method:
            acq_method = acq_method.split(':')[-1]
        dic_acqp["scan_method"] = acq_method

        #-------etc
        dic_acqp["scan_name"] = scan_acqp['ACQ_scan_name']
        dic_acqp["flip_angle"] = scan_acqp['ACQ_flip_angle']
        dic_acqp["obj_order"] = scan_acqp['ACQ_obj_order']
        dic_acqp["n_slices"] = scan_acqp['NSLICES']

        return dic_acqp

    # ==============================================================================
    #  get_dic_method: return scan method parameter
    # ==============================================================================
    def get_dic_method(self, scans, scan_num):
        """ Return scan method parameter

        :param scans:
        :param scan_num: Scan ID
        :return: TR, TE, BandWidth
        """

        scan_method = scans[str(scan_num)]['method']
        scan_pars = scans[str(scan_num)]['reco'][str(1)]

        # return
        dic_method = dict()

        #-------band width
        try:
            bw = scan_method['PVM_EffSWh'] / 1000.0
        except TypeError:
            bw = 0.0
        except Exception as e:
            if self.raise_exp:
                raise e
        dic_method["band_width"] = bw

        #-------etc
        try:
            dic_method["tr"] = scan_method['PVM_RepetitionTime']
            dic_method["te"] = scan_method['PVM_EchoTime']
        except:
            dic_method["tr"] = scan_pars['VisuAcqRepetitionTime']
            dic_method["te"] = scan_pars['VisuAcqEchoTime']

        dic_method[""] = scan_method['PVM_SPackArrGradOrient']
        dic_method[""] = scan_method['PVM_SPackArrReadOrient']
        dic_method["order_scheme"] = scan_method['PVM_ObjOrderScheme']
        dic_method["method"] = scan_method['Method']
        dic_method["naverages"] = scan_method['PVM_NAverages']
        dic_method["scan_time"] = scan_method['PVM_ScanTimeStr']
        try:
            dic_method["repetition"] = scan_method['PVM_NRepetitions']
        except:
            dic_method["repetition"] = 1

        if "PVM_EpiEchoSpacing" in scan_method:
            dic_method["echo_spacing"] = scan_method['PVM_EpiEchoSpacing']
        else:
            dic_method["echo_spacing"] = ""

        if "NSegments" in scan_method:
            dic_method["nsegments"] = scan_method['NSegments']

        return dic_method

    # ==============================================================================
    #  get_dic_reco: return scan recon parameter
    # ==============================================================================
    def get_dic_reco(self, scans, scan_num, reco_num):
        """ Return recon parameters.
        Calculate the information of the metrix size dimmension of the acqusition protocol, shape,
        number of slice packages, and index of slice_pack on shape will be returned.
        Calculate image resolution.

        :param scans:
        :param scan_num:
        :param reco_num:
        :return:
        """

        scan_method = scans[str(scan_num)]['method']
        scan_pars = scans[str(scan_num)]['reco'][str(reco_num)]
        scan_recoparam = scans[str(scan_num)]['recoparam'][str(reco_num)]

        # return dictionary
        dic_reco = dict()

        #------ visu_version
        if "VisuVersion" in scan_pars:
            visu_version = scan_pars['VisuVersion']
        else:
            visu_version = 0
        dic_reco["visu_version"] = visu_version

        #------ visu_subject_type
        if "VisuSubjectType" in scan_pars:
            dic_reco["visu_subject_type"] = scan_pars['VisuSubjectType']

        #------ dist
        if 'VisuCoreSlicePacksSliceDist' in scan_pars:
            dic_reco["dist"] = scan_pars['VisuCoreSlicePacksSliceDist']
        else:
            dic_reco["dist"] = scan_pars['VisuCoreFrameThickness']

        #------ scan time
        if visu_version == 1:
            dic_reco["visu_scan_time"] = scan_pars['VisuAcqScanTime']
            dic_reco["visu_acq_date"] = scan_pars['VisuAcqDate']
        else:
            dic_reco["visu_creat_date"] = scan_pars['VisuCreationDate']

        #------ dim
        dim = scan_pars['VisuCoreDim']
        dic_reco["dim"] = dim

        #------ matrix
        matrix = scan_pars['VisuCoreSize']
        if not isinstance(matrix, list):
            matrix = [matrix]
        dic_reco["matrix"] = matrix

        #------ slice_packs
        # Check number of slice packages
        if visu_version == 1:
            if isinstance(scan_method['PVM_SPackArrSliceOrient'], list):
                slice_packs = len(scan_method['PVM_SPackArrSliceOrient'])
            else:
                slice_packs = 1
        else:
            if 'VisuCoreSlicePacksDef' in scan_pars.keys():
                slice_packs = scan_pars['VisuCoreSlicePacksDef'][1]
            else:
                slice_packs = 1
        dic_reco["slice_packs"] = slice_packs

        #------ slices
        # Calculate number of slices
        if 'VisuCoreSlicePacksSlices' in scan_pars.keys():
            slices_param = scan_pars['VisuCoreSlicePacksSlices']
            if len(scan_pars['VisuCoreSlicePacksSlices']) > 1:
                if isinstance(slices_param[0], int):
                    slices = slices_param[1]
                else:
                    slices = slices_param[-1][-1]
            else:
                slices = len(scan_pars['VisuCoreOrientation'])
        else:
            if 'VisuCoreOrientation' in scan_pars.keys():
                slices = len(scan_pars['VisuCoreOrientation']) / slice_packs
            else:
                slices = 1
        dic_reco["slices"] = slices

        #------ frame
        # Add rest of frame as 4th dimension
        frame = scan_pars['VisuCoreFrameCount'] / slices / slice_packs
        dic_reco["frame"] = frame

        #------ shape
        # Complete 3D matrix shape
        shape = matrix[:]
        if slices == 1:
            if dim < 3:
                shape.append(slices)
        else:
            shape.append(slices)
        if frame > 1:
            shape.append(frame)
        dic_reco["shape"] = shape

        #------ idx
        if slice_packs > 1:
            shape.append(slice_packs)
            idx = len(shape) - 1
        else:
            idx = None
        dic_reco["idx"] = idx

        #------ tf
        # transform matrix saved on VisuPars
        # The orientation matrix VisuCoreOrientation describes
        # the conversion from the subject orientation system
        # into the image orientation system
        tfs = scan_pars['VisuCoreOrientation']
        tf = tfs == tfs[0, :]
        if tf.all():
            tf = tfs[0, :].reshape([3, 3])
        else:
            tf = tfs.reshape([tfs.shape[0], 3, 3])
        dic_reco["tf"] = tf

        # VisuCorePosition is the position of the first pixel
        # in the image in the subject coordinate system
        # (not the position of the middle pixel)
        dic_reco["visu_position"] = scan_pars['VisuCorePosition']

        # The field of view (VisuCoreExtent) is given in the image coordinate system.
        dic_reco["core_extent"] = scan_pars['VisuCoreExtent']

        # Data scaling: slope and offset
        dic_reco["scl_slope"] = scan_pars['VisuCoreDataSlope']
        dic_reco["scl_inter"] = scan_pars['VisuCoreDataOffs']

        #------ etc
        dic_reco["method_slice_orient"] = scan_method['PVM_SPackArrSliceOrient']
        dic_reco["bite_order"] = scan_pars['VisuCoreByteOrder']
        dic_reco["word_type"] = scan_pars['VisuCoreWordType']
        dic_reco["map_mode"] = scan_recoparam['RECO_map_mode']
        dic_reco["map_range"] = scan_recoparam['RECO_map_range']



        return dic_reco

    # ==============================================================================
    #  get_temp_resol
    # ==============================================================================
    def get_temp_resol(self, dic_method):
        """ Calculate temporal resolution

        :param dic_method:
        :return:
        """
        num_avr = dic_method['naverages']
        tr = dic_method['tr']
        if "nsegments" in dic_method:
            temp_resol = tr * dic_method['nsegments'] * num_avr
        else:
            temp_resol = tr * num_avr

        # self.log("[get_temp_resol] temp_resol ==[{}]".format(temp_resol))

        return temp_resol

    # def get_orient(self, scans, scan_num):
    #
    #
    #
    #     pattern = r'\<[+](.*);(.*)\>\s+\<[+](.*);(.*)\>\s+\<[+](.*);(.*)\>'
    #     direction = raw.scans['26']['method']['PVM_SliceGeo'][3]
    #     direction = dict(zip(re.sub(pattern, r'\2 \4 \6', direction).split(r' '),
    #                          re.sub(pattern, r'\1 \3 \5', direction).split(r' ')))

    # ==============================================================================
    #  get_center:
    # ==============================================================================
    def get_center(self, dic_reco, ret_val):
        """ Check if slice orientation is inverted

        :param dic_reco:
        :param ret_val: 1-center, 2-inverted
        :return:
        """

        slice_packs = dic_reco["slice_packs"]
        if slice_packs == 1:
            first = dic_reco["visu_position"][0]
            last = dic_reco["visu_position"][-1]
            # center = np.mean(dic_reco["visu_position"], axis=0)

            if first.sum() - last.sum() < 0:
                center = first
                inverted = False
            else:
                center = last
                inverted = True
        else:
            center = dic_reco["visu_position"]
            inverted = False

        if ret_val == 1:
            # self.log("# [get_center] center=[{}]".format(center))
            return center
        elif ret_val == 2:
            # self.log("# [get_center] inverted=[{}]".format(inverted))
            return inverted
        else:
            return ''

    # ==============================================================================
    #  get_code:
    # ==============================================================================
    def get_code(self, dic_reco):
        """ Check datatype and generate the code
        This function is immature, need to do more test in case Paravision 6 uses different data type
        (maybe for complex number?)

        :param dic_reco:
        :return:
        """

        bite_order = dic_reco['bite_order']
        if bite_order == 'littleEndian':
            code = '<'
        elif bite_order == 'bigEndian':
            code = '>'
        else:
            code = '='
        word_type = dic_reco['word_type'].split('_')[1:]
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

    # ==============================================================================
    #  get_resol:
    # ==============================================================================
    def get_resol(self, dic_reco):
        """ Calculate image resolution

        :param dic_reco:
        :return:
        """
        core_extent = dic_reco['core_extent'][:]
        orient = dic_reco['method_slice_orient']
        dist = dic_reco["dist"]
        dim = dic_reco["dim"]
        shape = dic_reco["shape"]
        tf = np.round(dic_reco["tf"], decimals=0)
        slice_packs = dic_reco["slice_packs"]

        if isinstance(dist, list):
            dist = dist[0]
        if dim == 2:
            core_extent.append(shape[2] * dist)

        # calculate resolution and correct axes
        try:
            resol = np.asarray(map(float, core_extent)) / np.asarray(shape[:3])
            if slice_packs == 1:
                resols = list(map(abs, np.dot(resol, tf)))
                # print('{0}: {1}\n{2}'.format(orient, list(map(abs, np.dot(resol, tf.T))), tf.T))
            else:
                resols = []
                for i in range(slice_packs):
                    if orient[i] == 'sagittal':
                        resols.append(map(abs, np.dot(resol, tf[i])))
                    else:
                        resols.append(map(abs, np.dot(resol, tf[i].T)))

                    # print('{0}: {1}\n{2}'.format(orient[i], np.dot(resol, tf[i]), tf[i]))

            return resols

        except Exception as e:
            if self.raise_exp:
                raise e

    # ==============================================================================
    #  get_scan_time: return list scan number
    # ==============================================================================
    def get_scan_time(self, subject, scans, item):
        """
        return scan date, scan start time and scan end time
        :param subject:
        :param scans:
        :param item: all- return date, start time, end time
                     date- return date
                     time/duration- return start time, end time
        :return:
        """
        last_scan = str(self.get_scan_list(scans)[-1])
        dic_reco = self.get_dic_reco(scans, last_scan, 1)

        if dic_reco["visu_version"] == 1:
            pattern = r'(\d{2}:\d{2}:\d{2})\s(\d{2}\s\w+\s\d{4})'

            # parsing and conversion type (str->date)

            # start time
            subject_date = subject['SUBJECT_date']
            start_time = datetime.time(*map(int, re.sub(pattern, r'\1', subject_date).split(':')))

            # date
            date = datetime.datetime.strptime(re.sub(pattern, r'\2', subject_date), '%d %b %Y').date()

            # end time
            last_scan_time = dic_reco['visu_acq_date']
            last_scan_time = datetime.time(*map(int, re.sub(pattern, r'\1', last_scan_time).split(':')))
            acq_time = dic_reco['visu_scan_time'] / 1000
            time_delta = datetime.timedelta(0, acq_time)
            end_time = (datetime.datetime.combine(date, last_scan_time) + time_delta).time()
        else:

            pattern = r'(\d{4}-\d{2}-\d{2})[T](\d{2}:\d{2}:\d{2})'

            # start time
            subject_date = subject['SUBJECT_date'].split(',')[0]
            start_time = datetime.time(*map(int, re.sub(pattern, r'\2', subject_date).split(':')))

            # date
            date = datetime.date(*map(int, re.sub(pattern, r'\1', subject_date).split('-')))

            # end date
            end_time = dic_reco['visu_creat_date'].split(',')[0]
            end_time = datetime.time(*map(int, re.sub(pattern, r'\2', end_time).split(':')))

        # return
        if item == 'all':
            return date, start_time, end_time
        elif item == 'date':
            return date
        elif item == 'time' or item == 'duration':
            return start_time, end_time

    # ==============================================================================
    #  get_img:
    # ==============================================================================
    def get_img(self, scan_num, reco_num, dic_reco):
        """ Reshape loaded data into image structure

        :param scan_num: Scan ID
        :param reco_num: Reco ID
        :param dic_reco
        :return: Shaped image
        """

        # Load Data
        code = self.get_code(dic_reco)
        # self.log("# [get_img] code == [{}]".format(code))

        data_path = os.path.join(self.path, str(scan_num), 'pdata', str(reco_num), '2dseq')
        data = np.fromfile(data_path, dtype=np.dtype(code))

        # Get Image
        shape = dic_reco['shape']
        img = data.reshape(shape[::-1]).T
        inverted = self.get_center(dic_reco, 2)
        # self.log("# [get_img] inverted={}".format(inverted))

        if inverted:
            img = img[:, :, ::-1, ...]
        return img

    def calc_affine(self, dic_reco, quadruped=True):
        """ Calculate affine transformation matrix

        :param dic_reco:
        :param quadruped: True if animal type is quadruped for correct position
        :return: Affine transformation matrix
        """

        visu_version = dic_reco["visu_version"]
        slice_packs = dic_reco["slice_packs"]
        tf = np.round(dic_reco["tf"], decimals=0)
        # tf = dic_reco["tf"]
        resol = self.get_resol(dic_reco)
        center = self.get_center(dic_reco, 1)

        # for PV5.1 and below
        if visu_version == 1:
            if quadruped:
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
            if dic_reco["visu_subject_type"] == 'Quadruped':  # for PV6 and above
                correct = np.array([[1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]])
            else:
                correct = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        # If image has only one slice package, return one affine matrix
        if slice_packs == 1:
            if isinstance(resol, list):
                affine = np.dot(np.diag(resol), tf.T)
                affine = np.append(affine.T, [center], axis=0).T
                affine = np.append(affine, [[0, 0, 0, 1]], axis=0)
                affine = np.dot(affine.T, np.diag([1, -1, 1, 1])).T
                affine = np.dot(affine.T, correct).T
            else:
                affine = None
        # If image has multiple slice packages, return list of affine matrices
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

    # ==============================================================================
    #  set_default_header:
    # ==============================================================================
    def set_default_header(self, nii, dic_method, dic_acqp):
        """ Update NifTi Header information

        :param nii: Nibabel NifTi Object
        :param dic_method
        :param dic_acqp
        :return: Nibabel NifTi Object with updated Header
        """
        nii.header.default_x_flip = False
        tr = self.get_temp_resol(dic_method)
        # self.log("# [set_default_header] tr =[{}]".format(tr))

        if re.search('EPI', dic_method['method'], re.IGNORECASE) and not re.search('DTI', dic_method['method'],
                                                                                   re.IGNORECASE):
            nii.header.set_xyzt_units('mm', 'sec')
            nii.header['pixdim'][4] = float(tr) / 1000
            nii.header.set_dim_info(slice=2)
            nii.header['slice_duration'] = float(tr) / (1000 * dic_acqp["n_slices"])
            list_slices = dic_acqp["obj_order"]
            if isinstance(list_slices, int):
                list_slices = [list_slices]
            if dic_method["order_scheme"] == 'User_defined_slice_scheme':
                nii.header['slice_code'] = 0
            elif dic_method["order_scheme"] == 'Sequential':
                nii.header['slice_code'] = 1
            elif dic_method["order_scheme"] == 'Reverse_sequential':
                nii.header['slice_code'] = 2
            elif dic_method["order_scheme"] == 'Interlaced':
                nii.header['slice_code'] = 3
            elif dic_method["order_scheme"] == 'Reverse_interlacesd':
                nii.header['slice_code'] = 4
            elif dic_method["order_scheme"] == 'Angiopraphy':
                nii.header['slice_code'] = 0
            if nii.header['slice_code'] == 1 or nii.header['slice_code'] == 3:
                nii.header['slice_start'] = max(list_slices)
                nii.header['slice_end'] = min(list_slices)
            else:
                nii.header['slice_start'] = min(list_slices)
                nii.header['slice_end'] = max(list_slices)
        else:
            nii.header.set_xyzt_units('mm', 'unknown')
            nii.header['qform_code'] = 1
            nii.header['sform_code'] = 0
        return nii


'''
##########################################################################################################
#  BrukerRaw
##########################################################################################################
'''


class BrukerRaw(BrukerRawBase):
    """ BrukerRaw class
    This class provides the function to import parameters, data from Bruker Raw data
    """
    # scanned, check_method, save_as, scan_datetime, summary

    # check_dt -> conversion_datatype
    # check_array -> conversion_array
    # check_version -> visu_version
    # check_method -> method
    # summary -> print_summary

    def __init__(self, path, summary=False, log_handler='', raise_exp=False):
        """ Initiate the object
        """

        self.path = path
        self.log_handler = log_handler
        self.raise_exp = raise_exp
        self.summary = summary

        # Super Class
        BrukerRawBase.__init__(self, self.log_handler, self.raise_exp)

        try:

            # Get 'subject' dictionary
            self.subject = self.get_raw_object(self.path, 'subject')

            # Get Version
            self.version = self.get_version(self.subject)

            # Get all scan information
            self.scans = self.get_scans()

            # Print Summary
            if summary:
                self.print_summary()

        except Exception as e:
            self.subject = None
            self.scans = None
            if raise_exp:
                raise e

    # ==============================================================================
    #  Properties
    # ==============================================================================
    @property
    def scanned(self):
        """ Return acquired scan numbers """
        return self.get_scan_list(self.scans)

    @property
    def study_id(self):
        """ Study ID """
        return self.subject['SUBJECT_study_nr']

    @property
    def subject_id(self):
        """ Subject ID """
        return self.subject['SUBJECT_id']

    @property
    def session_id(self):
        """ Session ID """
        return self.subject['SUBJECT_study_name']

    @property
    def subject_owner(self):
        """ OWNER """
        return self.subject['OWNER']

    @property
    def user_name(self):
        """ Name of Researcher """
        return self.subject['SUBJECT_name_string']

    @property
    def subject_type(self):
        """ Subject Type ex) Human """
        return self.subject['SUBJECT_type']

    @property
    def subject_gender(self):
        """ Subject Gender """
        return self.subject['SUBJECT_sex']

    @property
    def subject_weight(self):
        """ Subject's weight """
        return self.subject['SUBJECT_weight']

    @property
    def subject_position(self):
        """ Subject Position """
        return self.subject['SUBJECT_position']

    @property
    def subject_entry(self):
        """ Subject Entry """
        return self.subject['SUBJECT_entry']

    @property
    def subject_dob(self):
        """ Subject Date of Birth """
        return self.subject['SUBJECT_dbirth']

    # ==============================================================================
    #  get_scans:
    # ==============================================================================
    def get_scans(self):
        """ Check all scans and save the parameter into self.scans object as dictionary
        collecting information
        - method
        - acqp
        - visu_pars
        """
        scans = dict()

        for scan in os.listdir(self.path):
            scan_path = os.path.join(self.path, scan)

            if os.path.isdir(scan_path):
                scans[scan] = dict()

                try:
                    # method dictionary
                    scans[scan]['method'] = self.get_raw_object(scan_path, 'method')
                    # acqp dictionary
                    scans[scan]['acqp'] = self.get_raw_object(scan_path, 'acqp')

                    scans[scan]['reco'] = dict()
                    scans[scan]['recoparam'] = dict()

                    for reco in os.listdir(os.path.join(scan_path, 'pdata')):
                        reco_path = os.path.join(scan_path, 'pdata', reco)

                        if os.path.isdir(reco_path):
                            if 'visu_pars' in os.listdir(reco_path):
                                # visu_pars dictionary
                                scans[scan]['reco'][reco] = self.get_raw_object(reco_path, 'visu_pars')
                                # reco dictionary
                                scans[scan]['recoparam'][reco] = self.get_raw_object(reco_path, 'reco')
                except IOError:
                    del scans[scan]
                except Exception as e:
                    if self.raise_exp:
                        raise e

        return scans

    # ==============================================================================
    #  scan_datetime:
    # ==============================================================================
    def scan_datetime(self, item='all'):
        """ Return scanned date, start of scan time and end of scan time

        :param item: Multiple options
                - all : return date, start_time, end_time
                - date : return only date
                - time or duration : return start_time and end_time
        """
        return self.get_scan_time(self.subject, self.scans, item)

    # ==============================================================================
    #  print_summary: Print
    # ==============================================================================
    def print_summary(self, stdout=False):
        """ Print out brief summary of the raw data
        """
        if stdout is True:
            def log(str):
                print(str)
        else:
            log = self.log

        title = '\nParavision {}'.format(self.version)
        log(title)
        log('-' * len(title))
        log('\nYou are looking the summary of [{}]\n'.format(self.path))

        if len(self.scanned):

            date, start_time, end_time = self.scan_datetime()

            log('UserAccount:\t{}'.format(self.subject_owner))
            log('Researcher:\t{}'.format(self.user_name))
            log('Date:\t\t{}'.format(date))
            log('Scan duration:\t{} - {}'.format(start_time, end_time))

            log('Subject ID:\t{}'.format(self.subject_id))
            log('Session ID:\t{}'.format(self.session_id))
            log('Study ID:\t{}'.format(self.study_id))

            log('Subject Type:\t{}'.format(self.subject_type))
            log('Gender:\t\t{}'.format(self.subject_gender))
            log('Date of Birth:\t{}'.format(self.subject_dob))
            log('weight:\t\t{} kg'.format(self.subject_weight))
            log('Position:\t{}\t\tEntry:\t{}'.format(self.subject_position, self.subject_entry))

            log('\nScanID\tSequence::Protocol::[Parameters]')

            for scan_num in self.scanned:
                try:
                    dic_acqp = self.get_dic_acqp(self.scans, scan_num)
                    dic_method = self.get_dic_method(self.scans, scan_num)

                    params = "[ TR: {0}ms, TE: {1:.2f}ms, BW: {2:.2f}kHz, FlipAngle: {3} ]".format(dic_method["tr"],
                                                                                                   dic_method["te"],
                                                                                                   dic_method["band_width"],
                                                                                                   dic_acqp["flip_angle"])
                    log('{}:\t{}::{}::{}'.format(str(scan_num).zfill(3),
                                                 dic_acqp["scan_method"],
                                                 dic_acqp["scan_name"],
                                                 params))

                    for reco_num in self.scans[str(scan_num)]['reco'].keys():

                        dic_reco = self.get_dic_reco(self.scans, scan_num, reco_num)

                        dim = dic_reco['dim']
                        idx = dic_reco['idx']
                        size = dic_reco['shape']
                        map_range = dic_reco['map_range']

                        if dim > 1:
                            if idx:
                                del size[idx]
                                # resol = ','.join(['{}'.format(round(i, 2)) for i in dic_reco['resols'][0]])
                                resol = self.get_resol(dic_reco)
                                size = 'x'.join(map(str, size))
                                size = '{}, resol(mm): {}, slice_packs: {}'.format(size, resol, dic_reco['slice_packs'])
                            else:
                                # resol = ','.join(['{}'.format(round(i, 2)) for i in dic_reco['resols']])
                                resol = self.get_resol(dic_reco)
                                size = 'x'.join(map(str, size))
                                size = '{}, resol(mm): {}'.format(size, resol)

                        log('\t[{}] dim: {}D, size: {}, \n'
                            '\t     mapmode: {}, range: {} ~ {}'.format(str(reco_num).zfill(2), dim, size,
                                                                             dic_reco['map_mode'], map_range[0],
                                                                             map_range[1]))
                except:
                    pass

        else:
            log('Empty study...')

    # ==============================================================================
    #  save_as:
    # ==============================================================================
    def save_as(self, scan_num, reco_num, filename, path='./', ext='.nii.gz', absrange=False):
        """ Save image data into NifTi format

        :param scan_num: Scan ID
        :param reco_num: Reco ID
        :param filename: Output filename without extension
        :param path: directory (default: ./
        :param ext: extension (default: .nii.gz)
        :return: Boolean (if 1, image is not saved)
        """
        try:
            dic_acqp = self.get_dic_acqp(self.scans, scan_num)
            dic_method = self.get_dic_method(self.scans, scan_num)
            dic_reco = self.get_dic_reco(self.scans, scan_num, reco_num)

            dim = dic_reco['dim']
            idx = dic_reco['idx']
            slice_packs = dic_reco['slice_packs']

            filename = os.path.join(path, filename)

            if dim < 2:
                pass
            else:
                img = self.get_img(scan_num, reco_num, dic_reco)
                if absrange == True:
                    slope = dic_reco['slope']
                    if isinstance(slope, list):
                        slope = list(set(slope))[0]
                    img = img * float(slope)
                affine = self.calc_affine(dic_reco)
                if slice_packs == 1:
                    nii = nib.Nifti1Image(img, affine)
                    nii = self.set_default_header(nii, dic_method, dic_acqp)

                    r11, r12, r13 = affine[0, :3]
                    r21, r22, r23 = affine[1, :3]
                    r31, r32, r33 = affine[2, :3]
                    off_x, off_y, off_z = affine[3, :3]

                    a = 0.5 * np.sqrt(1 + r11 + r22 + r33)
                    b = 0.25 * (r32 - r23) / a
                    c = 0.25 * (r13 - r31) / a
                    d = 0.25 * (r21 - r12) / a

                    nii.header['qoffset_x'] = off_x
                    nii.header['qoffset_y'] = off_y
                    nii.header['qoffset_z'] = off_z
                    nii.header['quatern_b'] = b
                    nii.header['quatern_c'] = c
                    nii.header['quatern_d'] = d
                    nii.header['pixdim'][0] = -1
                    nii.header['pixdim'][1:4] = self.get_resol(dic_reco)
                    nii.to_filename('{}{}'.format(filename, ext))
                else:
                    img = np.swapaxes(img, idx, -1)
                    slice_orient = self.scans[str(scan_num)]['method']['PVM_SPackArrSliceOrient']
                    for i in range(slice_packs):
                        nii = nib.Nifti1Image(img[..., i], affine[i])
                        nii = self.set_default_header(nii, dic_method, dic_acqp)
                        nii.to_filename('{}_{}{}'.format(filename, slice_orient[i], ext))
                return 0
            return 1
        except:
            return 1

    def get_method(self, scan_num):
        return self.scans[str(scan_num)]['method']['Method']

