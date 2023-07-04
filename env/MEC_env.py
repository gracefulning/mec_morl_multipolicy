from copy import deepcopy
import numpy as np
import json


ACTION_TO_CLOUD = 0

RAYLEIGH_VAR = 1
RAYLEIGH_PATH_LOSS_A = 35
RAYLEIGH_PATH_LOSS_B = 133.6
RAYLEIGH_ANTENNA_GAIN = 0
RAYLEIGH_SHADOW_FADING = 8
RAYLEIGH_NOISE_dBm = -174

ZERO_RES = 1e-6
MAX_EDGE_NUM = 10
        
class MEC_Env():
    def __init__(self, conf_file='config1.json', conf_name='MEC_Config1', w=1.0, fc=None, fe=None, edge_num=None):
        #读配置文件
        config = json.load(open(conf_file, 'r'))
        param = config[conf_name]
        self.dt = param['dt']
        self.Tmax = param['Tmax']
        self.edge_num_L = param['edge_num_L']
        self.edge_num_H = param['edge_num_H']
        self.user_num = param['user_num']
        self.possion_lamda = param['possion_lamda']
        
        self.task_size_L = param['task_size_L']
        self.task_size_H = param['task_size_H']
        self.wave_cycle = param['wave_cycle']
        self.wave_peak = param['wave_peak']
        
        self.cloud_freq = param['cloud_cpu_freq']
        self.edge_freq = param['edge_cpu_freq']
        self.cloud_cpu_freq_peak = param['cloud_cpu_freq_peak']
        self.edge_cpu_freq_peak = param['edge_cpu_freq_peak']
        
        self.fc = fc
        self.fe = fe
        self.edge_n = edge_num
            
        
        self.cloud_C = param['cloud_C']
        self.edge_C = param['edge_C']
        self.cloud_k = param['cloud_k']
        self.edge_k = param['edge_k']
        self.cloud_off_power = param['cloud_off_power']
        self.edge_off_power = param['edge_off_power']
        
        self.cloud_user_dist_H = param['cloud_user_dist_H']
        self.cloud_user_dist_L = param['cloud_user_dist_L']
        self.edge_user_dist_H = param['edge_user_dist_H']
        self.edge_user_dist_L = param['edge_user_dist_L']
        
        self.cloud_off_band_width = param['cloud_off_band_width']
        self.edge_off_band_width = param['edge_off_band_width']
        self.noise_dBm = param['noise_dBm']
        self.reward_alpha = param['reward_alpha']
        self.w = w
        
        self.reset()
        
    
    def reset(self):
        self.step_cnt = 0
        self.task_size = 0
        self.task_user_id = 0
        self.step_cloud_dtime = 0
        self.step_edge_dtime = 0
        self.step_energy = 0
        self.rew_t = 0
        self.rew_e = 0
        self.arrive_flag = False
        self.invalid_act_flag = False
        self.cloud_off_list = []
        self.cloud_exe_list = []
        self.edge_off_lists = []
        self.edge_exe_lists = []
        self.unassigned_task_list = []
        self.action = ACTION_TO_CLOUD
        
        self.edge_num = np.random.randint(self.edge_num_L, self.edge_num_H+1)
        if self.edge_n:
            self.edge_num = self.edge_n
        self.action_space = self.edge_num + 1
        self.finish_time = np.array([0]*(self.edge_num+1))
        
        self.cloud_cpu_freq = np.random.uniform(self.cloud_freq-self.cloud_cpu_freq_peak, self.cloud_freq+self.cloud_cpu_freq_peak)
        self.cloud_cpu_freq = self.fc if self.fc else self.cloud_cpu_freq
        self.edge_cpu_freq = [0]*self.edge_num
        self.task_size_exp_theta = self.cloud_cpu_freq/self.cloud_C
        for i in range(self.edge_num):
            self.edge_cpu_freq[i] = np.random.uniform(self.edge_freq-self.edge_cpu_freq_peak, self.edge_freq+self.edge_cpu_freq_peak)
            self.edge_cpu_freq[i] = self.fe if self.fe else self.edge_cpu_freq[i]
            self.edge_off_lists.append([])
            self.edge_exe_lists.append([])
            self.task_size_exp_theta += self.edge_cpu_freq[i]/self.edge_C
        
        self.done = False
        self.reward_buff = []
        self.cloud_dist = np.random.uniform(self.cloud_user_dist_L, self.cloud_user_dist_H, size=(1, self.user_num))
        self.user_dist = self.cloud_dist
        for i in range(self.edge_num):
            edge_dist = np.random.uniform(self.edge_user_dist_L, self.edge_user_dist_H, size=(1, self.user_num))
            self.user_dist = np.concatenate((self.user_dist, edge_dist), axis=0)
            
        
        self.cloud_off_datarate, self.edge_off_datarate = self.updata_off_datarate()
        self.generate_task()
        return self.get_obs()
        
        
    def step(self, actions):
        assert self.done==False, 'enviroment already output done'
        self.step_cnt += 1
        self.step_cloud_dtime = 0
        self.step_edge_dtime = 0
        self.step_energy = 0
        finished_task = []
        
        #####################################################
        #分配任务
        if self.arrive_flag:
            assert actions <= self.edge_num and actions >= ACTION_TO_CLOUD ,'action not in the interval %d, %d'%(actions,self.edge_num)
            self.action = actions
            self.arrive_flag = False
            the_task = {}
            the_task['start_step'] =  self.step_cnt
            the_task['user_id'] = self.task_user_id
            the_task['size'] = self.task_size
            the_task['remain'] = self.task_size
            the_task['off_time'] = 0
            the_task['wait_time'] = 0
            the_task['exe_time'] = 0
            the_task['off_energy'] = 0
            the_task['exe_energy'] = 0
            
            if actions == ACTION_TO_CLOUD:
                the_task['to'] = 0
                the_task['off_energy'] = (the_task['size']/self.cloud_off_datarate[the_task['user_id']])*self.cloud_off_power
                the_task['exe_energy'] = the_task['size']*self.cloud_k*self.cloud_C*(self.cloud_cpu_freq**2)
                self.step_energy = the_task['off_energy'] + the_task['exe_energy']
                self.cloud_off_list.append(the_task)
            else:
                e = actions
                the_task['to'] = e
                the_task['off_energy'] = (the_task['size']/self.edge_off_datarate[e-1, the_task['user_id']])*self.edge_off_power
                the_task['exe_energy'] = the_task['size']*self.edge_k*self.edge_C*(self.edge_cpu_freq[e-1]**2)
                self.step_energy = the_task['off_energy'] + the_task['exe_energy']
                self.edge_off_lists[e-1].append(the_task)
        self.rew_t, self.rew_e = self.estimate_rew()
                
        #####################################################
        #产生到达任务
        self.generate_task()
        #####################################################
        #云网络
        #推进任务卸载与执行进度
        used_time = 0
        while(used_time<self.dt):
            off_estimate_time = []
            exe_estimate_time = []
            task_off_num = len(self.cloud_off_list)
            task_exe_num = len(self.cloud_exe_list)
            #估计卸载时间
            for i in range(task_off_num):
                the_user = self.cloud_off_list[i]['user_id']
                estimate_time = self.cloud_off_list[i]['remain']/self.cloud_off_datarate[the_user]
                off_estimate_time.append(estimate_time)
            #估计执行时间
            if task_exe_num > 0:
                cloud_exe_rate = self.cloud_cpu_freq/(self.cloud_C*task_exe_num)
            for i in range(task_exe_num):
                estimate_time = self.cloud_exe_list[i]['remain']/cloud_exe_rate
                exe_estimate_time.append(estimate_time)
            #运行（最短时间）
            if len(off_estimate_time)+len(exe_estimate_time) > 0:
                min_time = min(off_estimate_time + exe_estimate_time)
            else:
                min_time = self.dt
   
            run_time = min(self.dt-used_time, min_time)

            #推进卸载
            cloud_pre_exe_list = []
            retain_flag_off = np.ones(task_off_num, dtype=np.bool)
            for i in range(task_off_num):
                the_user = self.cloud_off_list[i]['user_id']
                self.cloud_off_list[i]['remain'] -= self.cloud_off_datarate[the_user]*run_time
                self.cloud_off_list[i]['off_energy'] += run_time*self.cloud_off_power
                # self.step_energy += run_time*self.cloud_off_power
                self.cloud_off_list[i]['off_time'] += run_time
                if self.cloud_off_list[i]['remain'] <= ZERO_RES:
                    retain_flag_off[i] = False
                    the_task = deepcopy(self.cloud_off_list[i])
                    the_task['remain'] = self.cloud_off_list[i]['size']
                    cloud_pre_exe_list.append(the_task)
            pt = 0
            for i in range(task_off_num):
                if retain_flag_off[i]==False:
                    self.cloud_off_list.pop(pt)
                else:
                    pt += 1
            #推进执行
            if task_exe_num > 0:
                cloud_exe_size = self.cloud_cpu_freq*run_time/(self.cloud_C*task_exe_num)
                cloud_exe_energy = self.cloud_k*run_time*(self.cloud_cpu_freq**3)/task_exe_num
            retain_flag_exe = np.ones(task_exe_num, dtype=np.bool)
            for i in range(task_exe_num):
                self.cloud_exe_list[i]['remain'] -= cloud_exe_size
                self.cloud_exe_list[i]['exe_energy'] += cloud_exe_energy
                self.cloud_exe_list[i]['exe_time'] += run_time
                if self.cloud_exe_list[i]['remain'] <= ZERO_RES:
                    retain_flag_exe[i] = False
            pt = 0
            for i in range(task_exe_num):
                if retain_flag_exe[i]==False:
                    self.cloud_exe_list.pop(pt)
                else:
                    pt += 1
            self.cloud_exe_list = self.cloud_exe_list + cloud_pre_exe_list
            used_time += run_time
        #####################################################
        #边缘网络
        for n in range(self.edge_num):
            #推进任务卸载与执行进度
            used_time = 0
            while(used_time<self.dt):
                off_estimate_time = []
                exe_estimate_time = []
                task_off_num = len(self.edge_off_lists[n])
                task_exe_num = len(self.edge_exe_lists[n])
                #估计卸载时间
                for i in range(task_off_num):
                    the_user = self.edge_off_lists[n][i]['user_id']
                    estimate_time = self.edge_off_lists[n][i]['remain']/self.edge_off_datarate[n,the_user]
                    off_estimate_time.append(estimate_time)
                #估计执行时间
                if task_exe_num > 0:
                    edge_exe_rate = self.edge_cpu_freq[n]/(self.edge_C*task_exe_num)
                for i in range(task_exe_num):
                    estimate_time = self.edge_exe_lists[n][i]['remain']/edge_exe_rate
                    exe_estimate_time.append(estimate_time)
                #运行（最短时间）
                if len(off_estimate_time)+len(exe_estimate_time) > 0:
                    min_time = min(off_estimate_time + exe_estimate_time)
                else:
                    min_time = self.dt

                run_time = min(self.dt-used_time, min_time)

                #推进卸载
                edge_pre_exe_list = []
                retain_flag_off = np.ones(task_off_num, dtype=np.bool)
                for i in range(task_off_num):
                    the_user = self.edge_off_lists[n][i]['user_id']
                    self.edge_off_lists[n][i]['remain'] -= self.edge_off_datarate[n,the_user]*run_time
                    self.edge_off_lists[n][i]['off_energy'] += run_time*self.edge_off_power
                    self.edge_off_lists[n][i]['off_time'] += run_time
                    if self.edge_off_lists[n][i]['remain'] <= ZERO_RES:
                        retain_flag_off[i] = False
                        the_task = deepcopy(self.edge_off_lists[n][i])
                        the_task['remain'] = self.edge_off_lists[n][i]['size']
                        edge_pre_exe_list.append(the_task)
                pt = 0
                for i in range(task_off_num):
                    if retain_flag_off[i]==False:
                        self.edge_off_lists[n].pop(pt)
                    else:
                        pt += 1
                #推进执行
                if task_exe_num > 0:
                    edge_exe_size = self.edge_cpu_freq[n]*run_time/(self.edge_C*task_exe_num)
                    edge_exe_energy = self.edge_k*run_time*(self.edge_cpu_freq[n]**3)/task_exe_num
                retain_flag_exe = np.ones(task_exe_num, dtype=np.bool)
                for i in range(task_exe_num):
                    self.edge_exe_lists[n][i]['remain'] -= edge_exe_size
                    self.edge_exe_lists[n][i]['exe_energy'] += edge_exe_energy
                    self.edge_exe_lists[n][i]['exe_time'] += run_time
                    if self.edge_exe_lists[n][i]['remain'] <= ZERO_RES:
                        retain_flag_exe[i] = False
                pt = 0
                for i in range(task_exe_num):
                    if retain_flag_exe[i]==False:
                        self.edge_exe_lists[n].pop(pt)
                    else:
                        pt += 1
                self.edge_exe_lists[n] = self.edge_exe_lists[n] + edge_pre_exe_list
                used_time += run_time

        #####################################################
        #done判定
        if (self.step_cnt >= self.Tmax):
            self.done = True
        done = self.done
        
        #####################################################
        #obs编码
        obs = self.get_obs()
        
        #####################################################
        #reward计算
        reward = self.get_reward(finished_task)
        
        #####################################################
        #备注信息
        info = {}
        return obs, reward, done ,info
    
    def generate_task(self):
        #####################################################
        #产生到达任务
        task_num = np.random.poisson(self.possion_lamda)
        for i in range(task_num):
            task = {}
            theta = self.task_size_exp_theta + self.wave_peak*np.sin(self.step_cnt*2*np.pi/self.wave_cycle)
            task_size = np.random.exponential(theta)
            task['task_size'] = np.clip(task_size, self.task_size_L, self.task_size_H)
            task['task_user_id'] = np.random.randint(0, self.user_num)
            self.unassigned_task_list.append(task)
            
        if self.step_cnt < self.Tmax:
            if len(self.unassigned_task_list) > 0:
                self.arrive_flag = True
                arrive_task = self.unassigned_task_list.pop(0)
                self.task_size = arrive_task['task_size']
                self.task_user_id = arrive_task['task_user_id']
            else:
                self.arrive_flag = True
                self.task_size = 0
                self.task_user_id = np.random.randint(0, self.user_num)
            
            
    def updata_off_datarate(self):
        rayleigh = RAYLEIGH_VAR/2*(np.random.randn(self.edge_num+1, self.user_num)**2 + np.random.randn(self.edge_num+1, self.user_num)**2)  
        path_loss_dB = RAYLEIGH_PATH_LOSS_A*np.log10(self.user_dist/1000) + RAYLEIGH_PATH_LOSS_B
        total_path_loss_IndB = RAYLEIGH_ANTENNA_GAIN - RAYLEIGH_SHADOW_FADING - path_loss_dB
        path_loss = 10**(total_path_loss_IndB/10)
        rayleigh_noise_cloud = 10**((RAYLEIGH_NOISE_dBm-30)/10)*self.cloud_off_band_width;
        rayleigh_noise_edge = 10**((RAYLEIGH_NOISE_dBm-30)/10)*self.edge_off_band_width;
        gain_ = (path_loss*rayleigh)
        cloud_gain = gain_[0,:]/rayleigh_noise_cloud
        edge_gain = gain_[1:,:]/rayleigh_noise_edge
        cloud_noise = 10**((self.noise_dBm-30)/10)*self.cloud_off_band_width;
        edge_noise = 10**((self.noise_dBm-30)/10)*self.edge_off_band_width;
        cloud_off_datarate = self.cloud_off_band_width*np.log2(1 + (self.cloud_off_power*(cloud_gain**2))/cloud_noise)  
        edge_off_datarate = self.edge_off_band_width*np.log2(1 + (self.edge_off_power*(edge_gain**2))/edge_noise)  
        return cloud_off_datarate, edge_off_datarate

    
    def get_obs(self):
        obs = {}
        
        servers = []
        cloud = []
        cloud.append(1)
        cloud.append(self.cloud_cpu_freq/1e9)
        cloud.append(self.edge_num)
        cloud.append(self.task_size/1e6)
        cloud.append(1-self.done)
        cloud.append(self.cloud_off_datarate[self.task_user_id]/1e6/100)
        cloud.append(len(self.cloud_exe_list))
        task_exe_hist = np.zeros([60])
        n = 0
        for task in self.cloud_exe_list:
            task_feature = int(task['remain']/1e6)
            if task_feature>=60:
                task_feature = 59
            task_exe_hist[task_feature] += 1
        cloud = np.concatenate([np.array(cloud), task_exe_hist], axis=0)
        servers.append(cloud)
        
        for ii in range(self.edge_num):
            edge = []
            edge.append(1)
            edge.append(self.edge_cpu_freq[ii]/1e9)
            edge.append(self.edge_num)
            edge.append(self.task_size/1e6)
            edge.append(1-self.done)
            edge.append(self.edge_off_datarate[ii,self.task_user_id]/1e6/100)
            edge.append(len(self.edge_exe_lists[ii]))
            task_exe_hist = np.zeros([60])
            n = 0
            for task in self.edge_exe_lists[ii]:
                task_feature = int(task['remain']/1e6)
                if task_feature>=60:
                    task_feature = 59
                task_exe_hist[task_feature] += 1
            edge = np.concatenate([np.array(edge), task_exe_hist], axis=0)
            servers.append(edge)
        
        obs['servers'] = np.array(servers).swapaxes(0,1)
        
        re = obs['servers']
        return re
    
    def estimate_rew(self):
        remain_list = []
        if self.action == ACTION_TO_CLOUD:
            for task in self.cloud_exe_list:
                remain_list.append(task['remain'])
            computing_speed = self.cloud_cpu_freq/self.cloud_C
            offload_time = self.task_size/self.cloud_off_datarate[self.task_user_id] if self.task_size>0 else 0
        else:
            for task in self.edge_exe_lists[self.action-1]:
                remain_list.append(task['remain'])
            computing_speed = self.edge_cpu_freq[self.action-1]/self.edge_C
            offload_time = self.task_size/self.edge_off_datarate[self.action-1][self.task_user_id] if self.task_size>0 else 0

        remain_list = np.sort(remain_list)
        
        last_size = 0
        t2 = 0
        task_num = len(remain_list)
        for i in range(task_num):
            size = remain_list[i]
            current_speed = computing_speed/(task_num-i)
            t2 += (task_num-i)*(size-last_size)/current_speed
            last_size = size
        
        last_size = 0
        t_norm = 0
        t1 = 0
        task_num = len(remain_list)
        for i in range(task_num):
            size = remain_list[i]
            current_speed = computing_speed/(task_num-i)
            use_t = (size-last_size)/current_speed
            if t_norm + use_t >= offload_time:
                t_cut = offload_time - t_norm
                t1 += (task_num-i)*t_cut
                t_norm = offload_time
                remain_list[i] -= t_cut*current_speed
                remain_list[i] = 0 if remain_list[i]<ZERO_RES else remain_list[i]
                remain_list = remain_list[i:]
                break
            else:
                t1 += (task_num-i)*(size-last_size)/current_speed
                t_norm += use_t
            last_size = size
        
        remain_list = remain_list.tolist()
        remain_list.append(self.task_size)
        remain_list = np.sort(remain_list)
        last_size = 0
        task_num = len(remain_list)
        for i in range(task_num):
            size = remain_list[i]
            current_speed = computing_speed/(task_num-i)
            t1 += (task_num-i)*(size-last_size)/current_speed
            last_size = size
        
        reward_dt = t1 - t2
        if self.task_size > 0:
            reward_dt = -reward_dt*0.01
            reward_de = -self.step_energy*50
        else:
            reward_dt = 0
            reward_de = 0
        
        return reward_dt, reward_de
    
    def get_reward(self, finished_task):

        reward = self.w*self.rew_t + (1.0-self.w)*self.rew_e
        
        return reward

    
    def rander(self):
        pass
    