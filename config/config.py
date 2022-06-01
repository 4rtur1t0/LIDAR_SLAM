"""
Main config file of video and camera parameters.
"""
import yaml

class ParametersConfig():
    """
    Clase en la que se almacenan los parametros del registration
    """
    def __init__(self, yaml_file='config/parameters.yml'):
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)
            self.folder_name = config.get('folder_name')
            self.directory = config.get('directory')

            self.voxel_size = config.get('down_sample').get('voxel_size')

            self.do_filter_by_distance = config.get('filter_by_distance').get('decision')
            self.max_distance = config.get('filter_by_distance').get('distance')

            self.filter_ground_plane = config.get('filter_ground_plane').get('decision')
            self.radius_gd = config.get('filter_ground_plane').get('radius_normals')
            self.max_nn_gd = config.get('filter_ground_plane').get('maximum_neighbors')

            self.radius_normals = config.get('normals').get('radius_normals')
            self.max_nn = config.get('normals').get('maximum_neighbors')

            self.point2 = config.get('icp').get('point2')
            self.distance_threshold = config.get('icp').get('distance_threshold')
            self.step = config.get('icp').get('step')

            self.vis_normals = config.get('visualization').get('normals')
            self.vis_registration = config.get('visualization').get('registration_result')

            self.exp_deltaxy = config.get('experiment').get('deltaxy')
            self.exp_deltath = config.get('experiment').get('deltath')
            self.exp_long = config.get('experiment').get('long')




PARAMETERS = ParametersConfig()
