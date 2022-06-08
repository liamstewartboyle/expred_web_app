import logging


def get_logger(name):
    extra = {'app_name': name}
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s [counterfactual]: %(message)s')
    syslog = logging.StreamHandler()
    syslog.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(syslog)
    logger = logging.LoggerAdapter(logger, extra)
    return logger
