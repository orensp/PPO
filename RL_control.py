import tensorflow as tf
from keras.layers import Input, Dense
import tensorflow_probability as tfp
import numpy as np
from robosuite.controllers.logz import *
import time
import os


from keras.models import load_model

class PPO(object):
    def __init__(
        self,
        Env,
        sess=None,
        exp_name="try",
        n_iter=100,
        horizon=50,
        paths=50,
        reward_to_go=True,
        nn_baseline=True,
        gama=0.99,
        n_layers_policy=2,
        size_layer_policy=64,
        policy_activation=tf.tanh,
        policy_output_activation=None,
        n_layers_value=2,
        size_layers_value=64,
        value_activation = tf.tanh,
        value_output_activation=None,
        e_clip=0.2,
        Load_nn=False,
        learning_rate=5e-4
    ):
        """"creating ppo object """
        #Env is the mujoco env child.
        self.env = Env
        #Env observation space
        self.obs_space=self.env.observation_space
        # Env action space
        self.action_space =self.env.dof
        # Tensor flow session.
        self.sess=sess
        #exp_name:
        self.exp_name=exp_name
        #log_dir
        self.logdir= exp_name + '_' + 'PPO' + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        self.logdir= os.path.join('data', self.logdir)
        # Number of iteration to run before backpropogation
        self.n_iter=n_iter
        #Number of paths to collect
        self.paths=paths
        #how far in the future to go
        self.horizon=horizon
        # Reward to go or not
        self.reward_to_go=reward_to_go
        # Use baseline to decrease variance
        self.nn_baseline=nn_baseline
        #discount factor:
        self.gama=gama
        # Number of layers policy
        self.n_layers_policy=n_layers_policy
        # number of neurons in a layer policy
        self.size_layer_policy= size_layer_policy
        # policy activation
        self.activation_policy = policy_activation
        # policy output activation
        self.output_activation_policy = policy_output_activation
        # Number of layers value
        self.n_layers_value = n_layers_value
        # number of neurons in a layer value
        self.size_layer_value= size_layers_value
        # value activation
        self.activation_value = value_activation
        # value output activation
        self.output_activation_value = value_output_activation
        # clipping the change in policy 0.2 value in the paper
        self.e_clip= e_clip
        #learning or Executing available policy
        self.load_nn=Load_nn
        #learning rate
        self.learning_rate = learning_rate

    def _build_mlp(
            self,
            input_placeholder,
            output_size,
            scope
    ):
        nn_name = 'policy' if not scope.find('policy') == -1 else 'value'

        [n_layers,size_layer,activation,output_activation]=[self.__dict__["n_layers_"+str(nn_name)],self.__dict__["size_layer_"+str(nn_name)],
                                                            self.__dict__["activation_"+str(nn_name)],self.__dict__["output_activation_"+str(nn_name)]]

        with tf.variable_scope(scope):
            x = Input(tensor=input_placeholder)
            for i in range(n_layers):
                x = Dense(size_layer, activation=activation, name='fc' + str(i))(x)
            output_placeholder = Dense(output_size, activation=output_activation, name=str(nn_name) + "_action")(x)

        return output_placeholder


    def _locating_tf(self):

        [ob_dim,ac_dim]=[self.obs_space,self.action_space]

        self.sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)

        self.sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)
        self.old_log_prob = tf.placeholder(shape=[None], name="old_prob", dtype=tf.float32)

            # Define a placeholder for advantages
        self.sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        self.reward = tf.placeholder(shape=[None], name="rew", dtype=tf.float32)


    def _discount_rewards_to_go(self,rewards):
        res = []
        future_reward = 0
        gamma=self.gama

        for r in reversed(rewards):
            future_reward = future_reward * gamma + r
            res.append(future_reward)
        # return the ^-1 list nice way:)
        return res[::-1]

    #dont take into acount causality:
    def _sum_discount_rewards(self,rewards):
        gamma = self.gama
        return sum((gamma ** i) * rewards[i] for i in range(len(rewards)))


    def _next_step_policy(self):
            sy_mean = self._build_mlp(
                input_placeholder=self.sy_ob_no,
                output_size=self.action_space,
                scope="policy_nn")

            # logstd should just be a trainable variable, not a network output??
            sy_logstd = tf.get_variable("logstd".format(np.random.rand(1)[0]), shape=[self.action_space])

            # random_normal just give me a number between -1 to 1. if we multiply this number by sigma we like sampling from
            # The normal distribution. sample=Mu+sigma*Z,Z~N(0,1)

            self.sy_sampled_ac = tf.math.add(sy_mean , tf.multiply(tf.exp(sy_logstd),tf.random_normal(tf.shape(sy_mean))),name="sampled_action")

            # Hint: Use the log probability under a multivariate gaussian.
            dist = tfp.distributions.MultivariateNormalDiag(loc=sy_mean, scale_diag=tf.exp(sy_logstd),name="myOutput")
            # caculate -log (PI(a))-we just need to enter a from our sampling along the paths.
            self.sy_logprob_n = -dist.log_prob(self.sy_ac_na,"log_prob_output")


    def _next_step_value(self):

        self.baseline_prediction = tf.squeeze(self._build_mlp(
            input_placeholder=self.sy_ob_no,
            output_size=1,
            scope="value_nn"))
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        self.baseline_target = tf.placeholder(shape=[None], dtype=tf.float32,name="baseline_target")

    def initiate_ppo_controler(self):
        if self.load_nn:
            tf.reset_default_graph()
        self._locating_tf()
        self._next_step_policy()
        self._next_step_value()
        self._initiate_session()

        self.baseline_loss = tf.losses.mean_squared_error(predictions=self.baseline_prediction, labels=self.baseline_target)

        #self.baseline_update_op=tf.get_default_graph.__dict__
        self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.baseline_loss)

        self.entropy_loss = -tf.reduce_sum(tf.multiply(tf.sigmoid(self.sy_sampled_ac), tf.log(tf.sigmoid(self.sy_sampled_ac))))
        self.critic_loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self.baseline_prediction, labels=self.reward))
        prob_ratio = tf.exp(self.sy_logprob_n - self.old_log_prob)
        clip_prob = tf.clip_by_value(prob_ratio, 1. - self.e_clip, 1. + self.e_clip)
        self.weighted_negative_likelihood = tf.multiply(self.sy_logprob_n, self.sy_adv_n)
        self.ppo_loss=tf.reduce_mean(tf.minimum(tf.multiply(prob_ratio,self.sy_adv_n), tf.multiply(clip_prob, self.sy_adv_n)))
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.ppo_loss + 0* self.entropy_loss)

        if not self.load_nn:
            self.sess.run(tf.global_variables_initializer())

        if self.load_nn:

            self.sess.run(tf.global_variables_initializer())
            path = os.path.join('NN_models', str(self.exp_name)+".meta")
            #path_meta = os.path.join(path, str(self.exp_name))
            saver = tf.train.import_meta_graph(path)
            saver.restore(self.sess,tf.train.latest_checkpoint('NN_models'))

            tvars = tf.trainable_variables()
            tvars_vals = self.sess.run(tvars)
            File_object2 = open("load.txt", "w")
            for var, val in zip(tvars, tvars_vals):
                File_object2.writelines(str(var.name) + "<---------->" + str(val))
            File_object2.close()


    def _initiate_session(self):

        if not self.sess :
            tf_config = tf.ConfigProto(device_count={"CPU": 8})
            self.sess = tf.Session(config=tf_config)
            self.sess.__enter__()






    def _V_future(self,V_s,steps):
        for i in steps:
            V_s[i - 1] = 0
        V_s = np.array(np.append(V_s, 0))
        return V_s[1:]

    def _pathlength(self,path):
        return len(path["reward"])

    def train(self):
        configure_output_dir(self.logdir)
        start = time.time()


        for itr in range(self.n_iter):

            print("********** Iteration %i ************ " % itr)
            paths=[]
            for num_path in range(self.paths):
                ob = self.env.reset()
                ob=np.concatenate((ob['robot-state'],ob['object-state']),axis=-1)
                obs,acs,rewards=[],[],[]
                steps = 0
                for t_horizon in range(self.horizon):
                    obs.append(ob)
                    ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no : ob[None, :]})
                    acs.append(ac[0,:])
                    ob, rew, done, _ = self.env.step(ac[0,:])
                    ob=np.concatenate((ob['robot-state'],ob['object-state']),axis=-1)
                    rewards.append(rew)
                    steps+=1
                    if done:
                        break

                path = {"observation": np.array(obs),
                        "reward": np.array(rewards),
                        "action": np.array(acs),
                        "steps": steps}

                paths.append(path)

            steps = [path["steps"] for path in paths]
            ob_no = np.concatenate([path["observation"] for path in paths])
            ac_na = np.concatenate([path["action"] for path in paths])
            log_old_temp = self.sess.run(self.sy_logprob_n, feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na})
            q_n=[]
            if self.reward_to_go:
                q_n = np.concatenate([self._discount_rewards_to_go(path["reward"]) for path in paths])
            else:
                q_n = np.concatenate(
                    [[self._sum_discount_rewards(path["reward"])] * self._pathlength(path) for path in paths])

            if self.nn_baseline and itr > 0:
                # ------------------------Caculate Advantage in baseline mode-----------------------
                b_n = self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: ob_no})
                b_n = (b_n - np.mean(b_n)) / np.std(b_n)
                b_n = np.mean(q_n) + b_n * np.std(q_n)
                adv_n = q_n - b_n
                # -----------------------------------Update baseline NN-----------------------------------
                scaled_q = (q_n - np.mean(q_n)) / np.std(q_n)
                self.sess.run(self.baseline_update_op, feed_dict={self.sy_ob_no: ob_no, self.baseline_target: scaled_q})
            else:
                adv_n = q_n.copy()

            #q_n = np.concatenate([path["reward"] for path in paths])
            #V_n = self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: ob_no})
            #V_n = (V_n - np.mean(V_n)) / np.std(V_n)
            #V_n_1 = self._V_future(V_n, steps)
            #y_t = q_n + self.gama * V_n_1
            #adv_n = y_t - V_n

            _, loss_value = self.sess.run([self.update_op, self.ppo_loss],
                                     feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na, self.sy_adv_n: adv_n,
                                                self.old_log_prob: log_old_temp})

            returns = [path["reward"].sum() for path in paths]
            ep_lengths = [self._pathlength(path) for path in paths]
            log_tabular("Time", time.time() - start)
            log_tabular("Iteration", itr)
            log_tabular("AverageReturn", np.mean(returns))
            log_tabular("StdReturn", np.std(returns))
            log_tabular("MaxReturn", np.max(returns))
            log_tabular("MinReturn", np.min(returns))
            log_tabular("EpLenMean", np.mean(ep_lengths))
            log_tabular("EpLenStd", np.std(ep_lengths))
            log_tabular("Loss", loss_value)
            dump_tabular()
            pickle_tf_vars()

    def render(self, env, n_iter=100):
        if env.has_renderer == False:
            raise ValueError("require has_renderer=True")
        else:
            ob = self.env.reset()
            ob = np.concatenate((ob['robot-state'],ob['object-state']),axis=-1)
            for iter in range(n_iter):
                ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: ob[None, :]})
                env.render()
                time.sleep(0.08)
                ob, rew, done, _ = env.step(ac[0, :])
                ob = np.concatenate((ob['robot-state'], ob['object-state']), axis=-1)
                if done:
                    break

    def save_tf_model(self):
        path = os.path.join('NN_models',str(self.exp_name))
        saver= tf.train.Saver(save_relative_paths=True)
        #temp=os.path.join('nn_weights', self.logdir)
        #tf.saved_model.simple_save(self.sess,temp,inputs={"ob":self.sy_ob_no},outputs={"myOutput":self.sy_logprob_n,"baseline_target":self.baseline_prediction})
        saver.save(self.sess,path)

        tvars = tf.trainable_variables()
        tvars_vals = self.sess.run(tvars)

        File_object = open("save.txt", "w")
        for var, val in zip(tvars, tvars_vals):
            File_object.writelines(str(var.name) + "<---------->" + str(val))
        File_object.close()

    def load_tf_model(self):
        path = os.path.join('NN_models', str(self.exp_name) + ".meta")
        self.sess=tf.Session()

        new_saver = tf.train.import_meta_graph(path)
        new_saver.restore(self.sess, tf.train.latest_checkpoint('NN_models/.'))

        #self.sess.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()
        self.sy_sampled_ac= graph.get_tensor_by_name("sampled_action:0")
        self.sy_ob_no =graph.get_tensor_by_name("ob:0")
        self.sy_ac_na = graph.get_tensor_by_name("ac:0")
        #dist = graph.get_tensor_by_name("myOutput:0")
        #self.sy_logprob_n=-dist.log_prob(self.sy_ac_na)

        tvars = tf.trainable_variables()
        tvars_vals = self.sess.run(tvars)
        File_object2 = open("load.txt", "w")
        for var, val in zip(tvars, tvars_vals):
            File_object2.writelines(str(var.name) + "<---------->" + str(val))
        File_object2.close()





































