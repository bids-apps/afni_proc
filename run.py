#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import subprocess
from glob import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import base64
import json
import numpy as np
import re

from io import open  # pylint: disable=W0622
import jinja2

__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'version')).read()

class Template(object):
    """
    Utility class for generating a config file from a jinja template.
    https://github.com/oesteban/endofday/blob/f2e79c625d648ef45b08cc1f11fd0bd84342d604/endofday/core/template.py
    """
    def __init__(self, template_str):
        self.template_str = template_str
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath='/'),
            trim_blocks=True, lstrip_blocks=True)

    def compile(self, configs):
        """Generates a string with the replacements"""
        template = self.env.get_template(self.template_str)
        return template.render(configs)

    def generate_conf(self, configs, path):
        """Saves the oucome after replacement on the template to file"""
        output = self.compile(configs)
        with open(path, 'w+') as output_file:
            output_file.write(output)


class IndividualTemplate(Template):
    """Specific template for the individual report"""

    def __init__(self):
        #super(IndividualTemplate, self).__init__(pkgrf('mriqc', 'data/reports/individual.html'))
        super(IndividualTemplate, self).__init__('/code/reports/individual.html')


class GroupTemplate(Template):
    """Specific template for the individual report"""

    def __init__(self):
        #super(GroupTemplate, self).__init__(pkgrf('mriqc', 'data/reports/group.html'))
        super(GroupTemplate, self).__init__('/code/reports/group.html')

def read_report_snippet(in_file):
    """Add a snippet into the report"""
    import os.path as op
    import re
    from io import open  # pylint: disable=W0622

    is_svg = (op.splitext(op.basename(in_file))[1] == '.svg')

    with open(in_file) as thisfile:
        if not is_svg:
            return thisfile.read()

        svg_tag_line = 0
        content = thisfile.read().split('\n')
        corrected = []
        for i, line in enumerate(content):
            if "<svg " in line:
                line = re.sub(' height="[0-9.]+[a-z]*"', '', line)
                line = re.sub(' width="[0-9.]+[a-z]*"', '', line)
                if svg_tag_line == 0:
                    svg_tag_line = i
            corrected.append(line)
        return '\n'.join(corrected[svg_tag_line:])


def make_montage(prefix, ulay=None, olay=None, cbar='FreeSurfer_Seg_i255',
                 opacity=4, montx=3, monty=1, blowup=1, delta_slices='-1 -1 -1',
                 func_range_perc=100):

    if ulay is None and olay is None:
        raise Exception("overlay and underlay can't both be undefined")
    elif ulay is None and olay is not None:
        ulay = olay
        olay = None

    cmd = '/code/@chauffeur_afni' + \
    ' -ulay ' + ulay
    if olay is not None:
        cmd += ' -olay ' + olay
        cmd += ' -set_dicom_xyz `3dCM {i}`'.format(i=olay)
        cmd += ' -cbar ' + cbar + \
               ' -opacity %d'%opacity 
    else:
        cmd += ' -olay_off'
        cmd += ' -set_dicom_xyz `3dCM {i}`'.format(i=ulay)
    cmd += ' -prefix ' + prefix + \
    ' -do_clean' + \
    ' -delta_slices '+ delta_slices + \
    ' -montx %d'%montx + \
    ' -monty %d'%monty + \
    ' -blowup %d'%blowup + \
    ' -func_range_perc %f' %func_range_perc + \
    ' -save_ftype JPEG'
    return cmd

def make_motion_plot(subj_dir, subj_id):
    # Read the three files in
    motion_file = os.path.join(subj_dir,'dfile_rall.1D')
    motion = pd.read_csv(motion_file, sep='\s*', engine = 'python', names = ['$\Delta$A-P [mm]','$\Delta$L-R [mm]','$\Delta$I-S [mm]','Yaw [$^\circ$]','Pitch [$^\circ$]','Roll [$^\circ$]'])
    
    enorm_file = os.path.join(subj_dir,'motion_{subj_id}_enorm.1D'.format(subj_id=subj_id))
    enorm = pd.read_csv(enorm_file, sep='\s*', engine = 'python', names = ['enorm'])

    outlier_file = os.path.join(subj_dir,'outcount_rall.1D')
    outliers = pd.read_csv(outlier_file, sep='\s*', engine = 'python', names = ['outliers'])
    
    # make a dataframe
    mot_df = pd.concat([outliers,enorm,motion], axis = 1)
    
    # Plot the dataframe
    axs = mot_df.plot(subplots = True, figsize = (4,5))
    ldgs = []
    for ax in axs:
        box = ax.get_position()
        ax.legend()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ldgs.append(ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)))
    plt.tight_layout()
    # save the figure
    qc_dir = os.path.join(subj_dir,'qc')
    img_dir = os.path.join(qc_dir,'img')
    if not os.path.exists(qc_dir):
        os.mkdir(qc_dir)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    out_path = os.path.join(img_dir,'motion_plot.svg')
    
    plt.savefig(out_path, tight_layout = True, bbox_extra_artists=ldgs, bbox_inches='tight')
    return out_path


def run(command, env={}, shell=False):
    merged_env = os.environ
    merged_env.update(env)
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=shell,
                               env=merged_env)
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() is not None:
            break
    if process.returncode != 0:
        raise Exception("Non zero return code: %d"%process.returncode)

task_re = re.compile('.*task-([^_]*)_.*')


parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
                    'should be stored. If you are running group level analysis '
                    'this folder should be prepopulated with the results of the'
                    'participant level analysis.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                    'Multiple participant level analyses can be run independently '
                    '(in parallel) using the same output_dir.'
                    'Only "participant" is currently supported.',
                    choices=['participant', 'group'])
parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                    'corresponds to sub-<participant_label> from the BIDS spec '
                    '(so it does not include "sub-"). If this parameter is not '
                    'provided all subjects should be analyzed. Multiple '
                    'participants can be specified with a space separated list.',
                    nargs="+")
parser.add_argument('--session_label', help='The label(s) of the sessions(s) that should be analyzed. The label '
                    'corresponds to ses-<session_label> from the BIDS spec '
                    '(so it does not include "ses-"). If this parameter is not '
                    'provided all sessions should be analyzed. Multiple '
                    'sessions can be specified with a space separated list.',
                    nargs="+")
parser.add_argument('--task_label', help='The label(s) of the tasks(s) that should be analyzed. The label '
                    'corresponds to task-<task_label> from the BIDS spec '
                    '(so it does not include "task-"). If this parameter is not '
                    'provided all tasks will be analyzed. Multiple '
                    'tasks can be specified with a space separated list.',
                    nargs="+")
parser.add_argument('--afni_proc', help='Optional: command string for afni proc. '
                    'Parameters that vary by subject '
                    'should be encapsulated in curly braces and must all be included '
                    '{{subj_id}}, {{out_dir}}, {{anat_path}}, or {{epi_paths}}.'
                    'The first _T1w for each subject will currently be used as the anat.'
                    'All of the _bold will be used as the functionals.'
                    'Example:'
                    '-subj_id {subj_id} '
                    '-scr_overwrite -out_dir {{out_dir}} '
                    '-blocks tshift align tlrc volreg blur mask scale '
                    '-copy_anat {{anat_path}} -tcat_remove_first_trs 0 '
                    '-dsets {{epi_paths}} -volreg_align_to MIN_OUTLIER '
                    '-volreg_align_e2a -volreg_tlrc_warp -blur_size 4.0 -bash')
parser.add_argument('--report_only', dest='report_only', action='store_true')
parser.add_argument('-v', '--version', action='version',
                    version='afni_proc BIDS-App {}'.format(__version__))


args = parser.parse_args()

bad_chars = ['`', '|', '&', ';', '>', '<', '$', '?', '(', ')', '\.', ':', '[', ']']

if args.afni_proc is not None:
    cmd_skeleton = args.afni_proc
    for bc in bad_chars:
        if bc in cmd_skeleton:
            raise Exception("Unsafe character '%s' found in command: %s"%(bc, cmd_skeleton))
    cmd_skeleton = 'python /opt/afni/afni_proc.py -check_results_dir no -script {ses_dir}/proc.bids.{subj_id}.{ses_id}.{task_id} '+ cmd_skeleton
else:
    cmd_skeleton = "python /opt/afni/afni_proc.py -check_results_dir no -subj_id {subj_id} \
-script {ses_dir}/proc.bids.{subj_id}.{ses_id}.{task_id} -scr_overwrite -out_dir {out_dir} \
-blocks tshift align tlrc volreg blur mask scale \
-copy_anat {anat_path} -tcat_remove_first_trs 0 \
-dsets {epi_paths}  -align_opts_aea -cost lpc+ZZ -giant_move \
-tlrc_base MNI152_T1_2009c+tlrc -tlrc_NL_warp \
-volreg_align_to MIN_OUTLIER \
-volreg_align_e2a -volreg_tlrc_warp -blur_size 4.0 -bash"""

run(('bids-validator %s'%args.bids_dir).split(' '))

# Get path for report directory
reports_dir = os.path.join(args.output_dir,"reports")


subjects_to_analyze = []
# only for a subset of subjects
if args.participant_label:
    subjects_to_analyze = args.participant_label[0].split(' ')
# for all subjects
else:
    subject_dirs = glob(os.path.join(args.bids_dir, "sub-*"))
    subjects_to_analyze = sorted([subject_dir.split("-")[-1] for subject_dir in subject_dirs])

# TODO: throw early error if they've specified participants, labels,
#  and subjects in such a way that there is nothing to analyze

# make sessions to analyze
# make tasks to analyze

all_configs = []
for subject_label in subjects_to_analyze:

    # get anatomical path
    anat_path = sorted(list(glob(os.path.join(args.bids_dir, "sub-%s"%subject_label,
                                           "anat", "*_T1w.nii*")) + glob(os.path.join(args.bids_dir,"sub-%s"%subject_label,"ses-*","anat", "*_T1w.nii*"))))[0]

    subj_out_dir = os.path.join(args.output_dir, "sub-%s"%subject_label)

    # Do sessions exist
    sessions_dirs = list(glob(os.path.join(args.bids_dir,"sub-%s"%subject_label,"ses-*")))
    sessions_list = [session_dir.split("-")[-1] for session_dir in sessions_dirs]
    if len(sessions_list) > 0:
        sessions_exist = True
        if args.session_label:
            sessions_to_analyze = sorted(set(args.session_label[0].split(' ')).intersection(set(sessions_list)))
        else:
            sessions_to_analyze = sessions_list
    else:
        sessions_exist = False 
        sessions_to_analyze = ['']
    
    for session_label in sessions_to_analyze:

        if sessions_exist:
            session_out_dir = os.path.join(subj_out_dir,"ses-%s"%session_label)

        else:
            session_out_dir = subj_out_dir
        os.makedirs(session_out_dir, exist_ok = True) 

        all_epi_paths = sorted(set(glob(os.path.join(args.bids_dir, "sub-%s"%subject_label,
                                                    "func", "*bold.nii*")) + glob(os.path.join(args.bids_dir,"sub-%s"%subject_label,"ses-%s"%session_label,"func", "*bold.nii*"))))

        # Which tasks to analyze
        try:
            tasks_in_session = set([task_re.findall(epi)[0] for epi in all_epi_paths])
        except:
            print("Tasks: ",[epi for epi in all_epi_paths if len(task_re.findall(epi))==0])
            raise Exception("A bold scan without a task label exists. Not permitted")
        if args.task_label:
            tasks_to_analyze = sorted(set(args.task_label[0].split(' ')).intersection(tasks_in_session))
        else:
            tasks_to_analyze = sorted(tasks_in_session)
            
        for task_label in tasks_to_analyze:
            epi_paths = ' '.join(sorted(set(glob(os.path.join(args.bids_dir, "sub-%s"%subject_label,
                                                    "func", "*%s*bold.nii*"%task_label)) + glob(os.path.join(args.bids_dir,"sub-%s"%subject_label,"ses-%s"%session_label,"func", "*%s*bold.nii*"%task_label)))))
            
            task_out_dir = os.path.join(session_out_dir,task_label)
            task_qc_dir = os.path.join(task_out_dir, 'qc')
            task_qc_img_dir = os.path.join(task_qc_dir, 'img')
        
            if args.analysis_level == 'participant':

                config = {}
                cmd = cmd_skeleton.format(subj_id=subject_label,ses_id = session_label, task_id = task_label, out_dir=task_out_dir,
                                          anat_path=anat_path, epi_paths=epi_paths, ses_dir = session_out_dir)
                if '{' in cmd:
                    raise Exception("Unsafe character '{' found in command: %s"%cmd.join(' '))
                cmd = cmd.replace('  ', ' ').split(' ')

                if not args.report_only:
                    print(' '.join(cmd), flush = True)
                    run(cmd)
                    print("tcsh -xef {ses_dir}/proc.bids.{subj_id}.{ses_id}.{task_id} 2>&1 | tee {ses_dir}/output.proc.bids.{subj_id}.{ses_id}.{task_id}".format(subj_id = subject_label,ses_id = session_label, task_id = task_label, ses_dir = session_out_dir), flush = True)
                    run("tcsh -xef {ses_dir}/proc.bids.{subj_id}.{ses_id}.{task_id}  2>&1 | tee {ses_dir}/output.proc.bids.{subj_id}.{ses_id}.{task_id}".format(subj_id = subject_label,ses_id = session_label, task_id = task_label, ses_dir = session_out_dir), shell=True)
                    run("mv  {ses_dir}/proc.bids.{subj_id}.{ses_id}.{task_id} {out_dir};mv  {ses_dir}/output.proc.bids.{subj_id}.{ses_id}.{task_id} {out_dir}".format(subj_id = subject_label,ses_id = session_label, task_id = task_label, ses_dir = session_out_dir, out_dir = task_out_dir), shell=True)

                pbs = glob(os.path.join(task_out_dir, 'pb*'))
                if len(pbs) > 0:
                    pb_lod = []
                    for pb in pbs:
                        pbd = {}
                        pbn = pb.split('/')[-1].split('.')
                        pbd['path'] = pb
                        pbd['filename'] = pb.split('/')[-1]
                        pbd['pb'] = int(pbn[0][-2:])
                        pbd['subj'] = pbn[1]
                        pbd['run'] = int(pbn[2][-2:])
                        pbd['block'] = pbn[3].split('+')[0]
                        pbd['orientation'] = pbn[3].split('+')[-1]
                        pb_lod.append(pbd)
                    pb_df = pd.DataFrame(pb_lod)
                    config['subj_id'] = pb_df.subj.unique()[0]
                    config['blocks'] = ' '.join(pb_df.block.unique())

                    try:
                        mot_path = make_motion_plot(task_out_dir, subject_label)
                        config['motion_report'] = read_report_snippet(mot_path)
                    except FileNotFoundError:
                        pass

                    warn_list = ['3dDeconvolve.err',
                                 'out.pre_ss_warn.txt',
                                 'out.cormat_warn.txt']

                    warns = {}
                    for wf in warn_list:
                        wf_path = os.path.join(task_out_dir, wf)
                        try:
                            if os.path.getsize(wf_path) > 0:
                                with open(wf_path, 'r') as h:
                                    warns[wf] = h.readlines()
                        except FileNotFoundError:
                            pass
                    if len(warns) > 0:
                        config['warnings'] = warns

                    if not os.path.exists(task_qc_dir):
                        os.mkdir(task_qc_dir)
                    if not os.path.exists(task_qc_img_dir):
                        os.mkdir(task_qc_img_dir)
                    if not os.path.exists(reports_dir):
                        os.mkdir(reports_dir)

                    try:

                        anat_path = os.path.join(task_out_dir, 'anat_final.%s+tlrc.HEAD'%subject_label)
                        anat_exts = np.array([float(ss) for ss in subprocess.check_output(["3dinfo", "-extent", anat_path]).decode().split('\t')])
                        anat_lrext = np.abs(anat_exts[0]) + np.abs(anat_exts[1])
                        anat_mont_dim = np.floor(np.sqrt(anat_lrext))
                        print("#######\n mont_dim = %f \n#########"%anat_mont_dim)
                        run(make_montage(os.path.join(task_qc_img_dir, 'anatomical_montage'),
                                         ulay=anat_path,
                                         montx=anat_mont_dim, monty=anat_mont_dim), shell=True)
                        

                        func_path = pb_df.loc[pb_df['block'] == 'volreg', 'path'].values[0] + '[0]'
                        func_rext = float(subprocess.check_output(["3dinfo", "-Rextent", func_path]))
                        func_lext = float(subprocess.check_output(["3dinfo", "-Lextent", func_path]))
                        func_lrext = np.abs(func_lext) + np.abs(func_rext)
                        func_mont_dim = np.floor(np.sqrt(func_lrext))

                        run(make_montage(os.path.join(task_qc_img_dir, 'functional_montage'),
                                         ulay=anat_path,
                                         olay=func_path, montx=anat_mont_dim, monty=anat_mont_dim,
                                         cbar='gray_scale', opacity=9), shell=True)

                        with open(os.path.join(task_qc_img_dir, 'anatomical_montage.sag.jpg'), 'rb') as h:
                            anat_bs = base64.b64encode(h.read()).decode()
                        with open(os.path.join(task_qc_img_dir, 'functional_montage.sag.jpg'), 'rb') as h:
                            func_bs = base64.b64encode(h.read()).decode()

                        config['volreg_report_anat'] = anat_bs
                        config['volreg_report_func'] = func_bs
                        config['anat_ap_ext'] = np.abs(anat_exts[2]) + np.abs(anat_exts[3]) + 1
                        config['anat_is_ext'] = np.abs(anat_exts[4]) + np.abs(anat_exts[5]) + 1
                        print("#######\n anat_ap_ext = %f \n#########"%config['anat_ap_ext'])
                    except (FileNotFoundError, ValueError):
                        pass

                    tpl = IndividualTemplate()
                    if sessions_exist:
                        tpl.generate_conf(config, os.path.join(reports_dir, 'sub-%s_ses-%s_task-%s_individual.html'%(subject_label, session_label, task_label)))
                    else:
                        tpl.generate_conf(config, os.path.join(reports_dir, 'sub-%s_task-%s_individual.html'%(subject_label, task_label)))

                    with open(os.path.join(task_qc_dir, 'individual.json'), 'w') as h:
                        json.dump(config, h)

            elif args.analysis_level == 'group':
                with open(os.path.join(task_qc_dir, 'individual.json'), 'r') as h:
                    all_configs.append(json.load(h))

if args.analysis_level == 'group':
    if not os.path.exists(reports_dir):
        os.mkdir(reports_dir)
    tpl = GroupTemplate()
    #print(all_configs)
    tpl.generate_conf({'configs':all_configs}, os.path.join(reports_dir, 'group.html'))
