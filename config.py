import configparser

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