#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, dyndep, model_helper, workspace

core.GlobalInit(['caffe2', '--caffe2_log_level=-1'])

dyndep.InitOpsLibrary("/home/galv/development/galvASR/build/galvASR/caffe2_ext/libcaffe2_ext.so")

if __name__ == '__main__':
    # arg_scope = {"order": "NCHW"}
    # train_model = model_helper.ModelHelper(name="asr_train", arg_scope=arg_scope)
    # dbreader = train_model.param_init_net.CreateDB([], 'dbreader_asr', db='scp:/home/galv/development/kws/kaldi/egs/mini_librispeech/s5/data/dev_clean_2/wav.scp', db_type='KaldiWaveDB')

    db = workspace.C.create_db('KaldiWaveDB', 'scp:/home/galv/development/kws/kaldi/egs/mini_librispeech/s5/data/dev_clean_2/wav.scp', workspace.C.Mode.read)
    cursor = db.new_cursor()
    from IPython import embed
    embed()
