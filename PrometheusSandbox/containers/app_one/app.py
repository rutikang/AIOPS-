from prometheus_client import start_http_server, Gauge, Histogram
import random
import time

g1 = Gauge('request_time_test', 'Random gauge 0 to 1')       
g2 = Gauge('request_time_train', 'Random gauge 0 to 0.6')


h = Histogram(name='request_latency_seconds', documentation='Histogram 0 - 1', buckets=(0.01,0.025,0.05,0.1,0.25,0.5,1,2.5,5,10,25) )
h2 = Histogram(name='request_latency_seconds2', documentation='Histogram 0 - 0.6', buckets=(0.01,0.025,0.05,0.1,0.25,0.5,1,2.5,5,10,25) )
# h2 = Histogram(name='request_latency_seconds', documentation='Histogram 0 - 0.6', buckets=(0.01,0.025,0.05,0.1,0.25,0.5,1,2.5,5,10,25) )


def emit_data(t):
    """Emit fake data"""
    time.sleep(t)
    g1.set(t)
    h.observe(t)
    
def emit_data2(t2):
    """Emit fake data"""
    time.sleep(t2)
    g2.set(t2)
    h2.observe(t2)
    


if __name__ == '__main__':
    start_http_server(8000)
    while True:
        emit_data(random.random())
        emit_data2(random.uniform(0, 0.6))
