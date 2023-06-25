import os
import json


class Properties:
    """
    The algorithm parameters.
    """

    def __init__(self, param_mat_name: str = 'default', para_scheme: int = 0):

        # check whether the JSON file exists
        config_name = '../configuration/config.json'
        assert os.path.exists(config_name), 'Config file is not accessible.'
        # open json
        with open(config_name) as f:
            cfg = json.load(f)['masp']

        self.cross_val = cfg['common']['crossValidation']
        self.cross_validate_num = cfg['common']['crossNum']
        self.learning_rate = cfg['common']['learningRate']
        self.budget = cfg['common']['budget']
        self.train_data_matrix = cfg['common']['trainDataMatrix']
        self.test_data_matrix = cfg['common']['testDataMatrix']
        self.train_label_matrix = cfg['common']['trainLabelMatrix']
        self.test_label_matrix = cfg['common']['testLabelMatrix']
        self.dc = cfg['common']['dc']
        self.pretrain_rounds = cfg['common']['pretrainRounds']
        self.increment_rounds = cfg['common']['incrementRounds']
        self.enhancement_threshold = cfg['common']['enhancementThreshold']
        self.cold_start_labels_proportion = cfg['common']['coldStartLabelsProportion']
        self.num_instances = cfg['common']['numInstances']
        self.num_conditions = cfg['common']['numConditions']
        self.num_labels = cfg['common']['numLabels']
        self.num_instances = cfg['common']['numInstances']
        self.num_conditions = cfg['common']['numConditions']
        self.num_labels = cfg['common']['numLabels']
        self.learning_rate = cfg['common']['learningRate']
        self.para_r= cfg['common']['para_r']
        self.mobp = cfg['common']['mobp']

        assert param_mat_name in cfg.keys(), "".join(
            ['The parameters of ', param_mat_name, 'are not defined in the JSON file of config.'])
        temp_dataset_cfg = cfg[param_mat_name]
        self.filename = temp_dataset_cfg['fileName']
        self.outputfilename = temp_dataset_cfg['outputFileName']

        assert os.path.exists(self.filename), 'Dataset file is not accessible.'
        self.parallel_layer_num_nodes = temp_dataset_cfg['parallelLayerNumNodes']
        self.para_r=temp_dataset_cfg['para_r']
        if para_scheme == 1:
            temp_array = [0, 2]
            temp_array[0] = self.parallel_layer_num_nodes[0]
            self.parallel_layer_num_nodes = temp_array

        self.activators = temp_dataset_cfg['activators']
        # get from data
        self.full_connect_layer_num_nodes = temp_dataset_cfg['fullConnectLayerNumNodes']

        if para_scheme == 2:
            temp_array2 = [0, 0]
            temp_array2[-1] = self.full_connect_layer_num_nodes[-1]
            self.full_connect_layer_num_nodes = temp_array2

        # two stage active learning
        self.label_batch = temp_dataset_cfg['labelBatch']
        self.instance_batch = temp_dataset_cfg['instanceBatch']
        self.instance_selection_proportion = temp_dataset_cfg['instanceSelectionProportion']
