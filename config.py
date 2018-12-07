import configparser
import json

def get_configs(file_name, name, key):
    """Get configurations from configuration file

    Arguments:
        file_name: String, configuration file name
        name: String, configuration name
        key: String, configuretion key

    Returns: 
        (name, key): Tuples, configuration name and value 
    """


    conf = configparser.ConfigParser()
    try:
        conf.read(file_name)
    except:
        print(file_name + 'not exists or no configuration of ' + name)

    return conf.get(name, key)

def get_network(file_name):
    """Get network architecture from file

    Arguments:
        file_name: String, network file name

    Returns:
        layers: List, network layers
        weights: Dictionary, network weights for all layers
        biases: Dictionary, network biases for all layers
    """


    with open(file_name, 'r') as f:
        model = json.load(f)
        layers = model['layers']
        weights = model['weights']
        biases = model['biases']

        return layers, weights, biases