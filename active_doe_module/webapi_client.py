import requests, time, os, sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

is_terminal = sys.stdout.isatty()
if is_terminal:
    try:
        from colorama import init
        from colorama import Fore
        init(autoreset=True)        
    except:
        is_terminal=False

if not is_terminal:
    class DummyFore(object):
        def __getattribute__(self, name: str):return ""
    Fore = DummyFore()


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token
    def __call__(self, r):
        r.headers["Authorization"] = "Bearer " + self.token
        return r

class active_doe_client(object):
    def __init__(self, hostname:str, port=9731, use_sg=True) -> None:
        _port = f":{port}" if port>0 else ""
        protocol="https" if use_sg else "http"
        relative_url='/active_doe' if use_sg else ''
        self.sg_url=f'{protocol}://{hostname}{_port}'
        self.base_url=self.sg_url+relative_url
        self.silent=False
        self.req_session = requests.Session()
        self.req_session.verify=False
        self.req_session.auth=None
        self.session_id=None
        self.use_sg=use_sg

    def __do_login(self):
        user=dict(
            UserName="cameo",
            Password="cameo"
        )
        with self.req_session.post(f'{self.sg_url}/login',json=user) as res:
            if res.status_code == 200:
                token=res.json()
                self.req_session.auth=BearerAuth(token["access_token"])
            else:
                print(f"{Fore.RED}failed to login{Fore.RESET}")
                raise Exception("failed to login")


    def ping_service(self):
        ping_url=self.base_url + "/services/is_alive"
        return self._ping(ping_url=ping_url)

    def ping_server(self):
        if not self.use_sg:
            return True
        else:
            ping_url=self.sg_url + "/ping"
            return self._ping(ping_url=ping_url)

    def _ping(self, ping_url):
        print(f"{Fore.CYAN}try pinging server at {ping_url}{Fore.RESET}")
        for nTry in range(10):
            try:
                with self.req_session.get(ping_url) as res:
                    if res.status_code == 200:
                        print(f"{Fore.CYAN}success{Fore.RESET}")
                        return True
            except:
                print(f"{Fore.YELLOW}Failed. Try a second later{Fore.RESET}")
            time.sleep(1)
        raise Exception(f"could not ping the server after {nTry} attempts")


    def initialize(self,setup):
        try:
            if self.ping_server():
                if self.use_sg:
                    self.__do_login()
                    if not self.ping_service():
                        return None
        except Exception as ex:
            print(f"{Fore.RED}{ex}{Fore.RESET}")
            return None
        response = self.req_session.post(url=self.base_url+'/services', json=setup)
        if not self.silent:
            message='---------------- initialization ----------------'
            print(f"{Fore.CYAN}{message}{Fore.RESET}")
        if response.status_code == 403:
            message = response.text
            print(f"{Fore.RED}{message}{Fore.RESET}")
            return None
        self.session_id=response.json()
        self.variation_names=[v["Name"] for v in setup["Variations"]]
        self.response_names=[v["Name"] for v in setup["Responses"]]
        return self.session_id

    def __send(self,method, route, **kwargs):
        rt = '/'+route if route else ''
        return getattr(self.req_session,method)(url=f'{self.base_url}/services/{self.session_id}{rt}', **kwargs)

    def write_samples(self, file_path, size=1, latest_models_required=True, server_target=False):
        if server_target:
            params={'size': size, 'latest_models_required': latest_models_required, 'file_path': file_path}
            return self.lengthy_request(params=params, method='post', route='samples')
        else:
            samples=self.get_samples(size=size, latest_models_required=latest_models_required)
            if samples is not None and len(samples)>0:
                v_names=[v for v in samples[0].get("Variations",{})]
                r_names=[v for v in samples[0].get("Responses",{})]
                data={v:[] for v in v_names+r_names}
                for s in samples:
                    for v in v_names: data[v].append(s["Variations"][v])
                    for v in r_names: data[v].append(s["Responses"][v])
                s=pd.DataFrame(data=data, index=range(len(samples)))
                s.to_csv(path_or_buf=file_path)
                return

    def get_samples(self, size=1, latest_models_required=True, retries=5):
        params={'size': size, 'latest_models_required': latest_models_required}
        samples=self.lengthy_request(params=params, method='get', route='samples')
        if samples is None or type(samples)==str:
            print(f"{Fore.RED}Did not receive valid samples. Got response '{samples}'{Fore.RESET}")
            if retries==0: return None
            else: return self.get_samples(size=size, latest_models_required=latest_models_required, retries=retries-1)
        if not self.silent:
            message=f'---------------- got {len(samples)} samples ----------------'
            print(f"{Fore.CYAN}{message}{Fore.RESET}")
            for i,c in enumerate(samples):
                if i>10:
                    print("...")
                    break
                print(c)
        return samples

    def get_candidates(self, size=1, latest_models_required=True, retries=5):
        params={'size': size, 'latest_models_required': latest_models_required}
        candidates=self.lengthy_request(params=params, method='get', route='candidates')
        if candidates is None or type(candidates)==str:
            print(f"{Fore.RED}Did not receive valid candidates. Got response '{candidates}'{Fore.RESET}")
            if retries==0: return None
            else: return self.get_candidates(size=size, latest_models_required=latest_models_required, retries=retries-1)
        if not self.silent:
            message='---------------- get candidates for iterations {} ----------------'.format([c['Index'] for c in candidates])
            print(f"{Fore.CYAN}{message}{Fore.RESET}")
            for i,c in enumerate(candidates):
                if i>10:
                    print("...")
                    break
                print('* Variations: {}'.format(c['Variations']))
            print(f"{Fore.MAGENTA}* ModelQuality: {candidates[-1]['Panel']['Modeling']['Quality']}{Fore.RESET}")
            print(f"{Fore.MAGENTA}* ModelCPUTime: {candidates[-1]['Panel']['Modeling']['CPUTime']}{Fore.RESET}")
            color = Fore.GREEN if candidates[-1]['Panel']['Algorithm']['StopRecommended'] else Fore.YELLOW
            print(f"{color}* StopMessage: {candidates[-1]['Panel']['Algorithm']['StopMessage']}{Fore.RESET}")
        return candidates

    def get_abs_url(self, abs_or_rel_url: str):
        if not abs_or_rel_url.startswith("http"):
            return f"{self.base_url}/services/{self.session_id}/{abs_or_rel_url}"
        else:
            return abs_or_rel_url

    def get(self, url, **kwargs):
        url=self.get_abs_url(url)
        result=self.req_session.get(url=url, **kwargs)
        status_code=result.status_code
        if status_code == 308:
            url=result.headers["Location"]
            return self.get(url=url, **kwargs)
        else:
            return result

    def lengthy_request(self, params={}, method='get', route='candidates'):
        points=None
        try:
            result=self.__send(method=method, route=route, params=params)
            if result.status_code==202 and 'Location' in result.headers:
                status_code=200
                while status_code == 200:
                    time.sleep(.25)
                    status=self.get(url=result.headers['Location'], allow_redirects=False)
                    status_code=status.status_code
                if status_code == 303:
                    result=self.get(url=status.headers['Location'], allow_redirects=False)
            if result.status_code!=200:
                raise Exception(f"Received unexpected server response {result.status_code}!=200 at location '{result.url}'")
            points=result.json()
        except Exception as e:
            print(f"{Fore.RED}{str(e)}{Fore.RESET}")
        return points

    def insert_measurements(self, measurements):
        try:
            if not type(measurements)==list:
                measurements=[measurements]
            if not self.silent:
                message='---------------- insert measurements for candidates {} ----------------'.format([m['Index'] if 'Index' in m else None for m in measurements])
                print(f"{Fore.CYAN}{message}{Fore.RESET}")
                for meas in measurements:
                    print('* Responses: {}'.format(meas['Responses']))
            for meas in measurements:
                for r in meas['Responses']:
                    if meas['Responses'][r] is None or np.isnan(meas['Responses'][r]):
                        meas['Responses'][r]=None
            self.__send('put','measurements',  json=measurements)
        except Exception as e:
            print(f"{Fore.RED}{str(e)}{Fore.RESET}")
            return

    def get_result(self):
        resp=self.__send('get','results')
        if resp.status_code==200:
            return resp.json()
        return None

    def load_start_data(self, file_path, server_target=False):
        if server_target:
            body=dict(
                FilePath=file_path,
            )
            resp = self.__send('put', 'results',  json=body)
            return resp.status_code==200
        
        if not os.path.isfile(file_path):
            print(f"{Fore.RED}File {file_path} does not exist!{Fore.RESET}")
            return False

        data=pd.read_csv(file_path, dtype=float)
        n_data=data.shape[0]
        for v in self.variation_names+self.response_names:
            if v not in data:
                print(f"{Fore.RED}File {file_path} does not contain data for channel {v}!{Fore.RESET}")
                return False
        measurements=[]
        for n in range(n_data):
            measurements.append(dict(
                Variations={v: data[v][n] for v in self.variation_names},
                Responses={v: data[v][n] for v in self.response_names},
            ))
        self.insert_measurements(measurements=measurements)
        return True

    def write_result(self, file_path, level=2, server_target=False):
        if not server_target:
            try:
                result=self.get_result()
                if result is None: return
                n=len(result["Index"])
                data={}
                for m in ["Variations", "Responses", "Measurements"]:
                    for v in result[m]:
                        data[v]=result[m][v]
                s=pd.DataFrame(data=data, index=range(n))
                s.to_csv(path_or_buf=file_path)
            except Exception as e:
                print(f"{Fore.RED}An error occured while writing the result file{Fore.RESET}")
                print(f"{Fore.RED}{str(e)}{Fore.RESET}")
        else:
            body=dict(
                FilePath=file_path,
                Level=level
            )
            self.__send('post', 'results',  json=body)

        if not self.silent:
            message='---------------- write result file ----------------'
            print(f"{Fore.CYAN}{message}{Fore.RESET}")
            color=Fore.GREEN if os.path.exists(file_path) else Fore.RED
            print(f"{color}* ResultFile: {file_path}{Fore.RESET}")

    def terminate(self):
        if self.session_id:
            if not self.silent:
                message='---------------- termination ----------------'
                print(f"{Fore.CYAN}{message}{Fore.RESET}")
            self.__send('delete',None)
            self.session_id=None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.terminate()
