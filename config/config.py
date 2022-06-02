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
            self.max_distance = config.get('filter_by_distance').get('distance')

            self.radius_gd = config.get('filter_ground_plane').get('radius_normals')
            self.max_nn_gd = config.get('filter_ground_plane').get('maximum_neighbors')

            self.radius_normals = config.get('normals').get('radius_normals')
            self.max_nn = config.get('normals').get('maximum_neighbors')

            self.fpfh_threshold = config.get('fpfh').get('fpfh_threshold')
            self.max_radius_descriptor = config.get('descriptor').get('max_radius_descriptor')
            self.distance_threshold = config.get('icp').get('distance_threshold')

            self.exp_deltaxy = config.get('experiment').get('deltaxy')
            self.exp_deltath = config.get('experiment').get('deltath')
            self.exp_long = config.get('experiment').get('long')




PARAMETERS = ParametersConfig()
