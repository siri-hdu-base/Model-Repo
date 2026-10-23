"""隔离推理进程：超时/超内存可终止，失败不能冒充正常预测。"""
import multiprocessing as mp
import time
import traceback
import numpy as np
import psutil


def worker(connection):
    from algorithms.registry import predict
    from algorithms.paper_temporal import RequiredPrior
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(1)
    while True:
        try:job=connection.recv()
        except EOFError:return
        if job is None:return
        name,x,profile,params,seed=job
        started=time.perf_counter()
        try:
            with threadpool_limits(limits=1):
                labels=np.asarray(predict(name,x,profile,params,seed),dtype=np.int64)
            if labels.shape!=(len(x),):raise ValueError("预测长度不等于输入脉冲数")
            connection.send({"status":"ok","labels":labels,"inference_seconds":time.perf_counter()-started})
        except RequiredPrior as e:
            connection.send({"status":"not_evaluated","reason":str(e)})
        except MemoryError:
            connection.send({"status":"memory_limit","reason":"算法内存不足"})
        except Exception as e:
            status="resource_limit" if str(e).startswith("resource_limit:") else "error"
            connection.send({"status":status,"reason":str(e),"traceback":traceback.format_exc()})


class IsolatedRunner:
    def __init__(self,timeout=60,memory_mb=4096,max_pulses=20000):
        self.timeout=timeout;self.memory_mb=memory_mb;self.max_pulses=max_pulses;self.process=None;self.connection=None
    def close(self):
        if self.process:
            if self.process.is_alive():
                self.process.terminate();self.process.join(5)
                if self.process.is_alive():self.process.kill();self.process.join()
            self.process.close();self.process=None
        if self.connection:self.connection.close();self.connection=None
    def run(self,name,x,profile,params,seed):
        if len(x)>self.max_pulses:
            return {"status":"resource_limit","reason":f"窗口 {len(x)} 脉冲超过显式上限 {self.max_pulses}；未抽样"}
        if self.process is None or not self.process.is_alive():
            self.close();parent,child=mp.get_context("spawn").Pipe()
            self.connection=parent;self.process=mp.get_context("spawn").Process(target=worker,args=(child,));self.process.start();child.close()
        wall=time.perf_counter();self.connection.send((name,x,profile,params,seed));peak=0;handle=psutil.Process(self.process.pid)
        while not self.connection.poll(.02):
            if not self.process.is_alive():
                self.close();return {"status":"worker_crash","reason":"独立推理进程异常退出"}
            try:peak=max(peak,handle.memory_info().rss/2**20)
            except psutil.Error:pass
            if peak>self.memory_mb:
                self.close();return {"status":"memory_limit","reason":"进程 RSS 超过配置上限","peak_rss_mb":peak}
            if time.perf_counter()-wall>self.timeout:
                self.close();return {"status":"timeout","reason":f"超过 {self.timeout} 秒","peak_rss_mb":peak}
        result=self.connection.recv()
        try:peak=max(peak,handle.memory_info().rss/2**20)
        except psutil.Error:pass
        result["peak_rss_mb"]=peak;result["wall_seconds"]=time.perf_counter()-wall
        if result["status"]=="ok":result["pulses_per_second"]=len(x)/max(result["inference_seconds"],1e-12)
        return result
