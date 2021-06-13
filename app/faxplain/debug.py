def get_bogus_pred():
    pred = {
        'clses': ['REFUTE' for i in range(3)],
        'evis': [[1 for j in range(3)] for i in range(3)],
        'links': ['www.abs.com' for i in range(3)],
        'query': 'this is not a query'
    }
    return pred