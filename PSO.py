'''
Reference: https://blog.csdn.net/weixin_44791964/article/details/98180528?spm=1001.2014.3001.5501
初始化粒子群
while 未达到最大迭代次数或最小loss：
	for each_p in 粒子群:
		计算适应度
		如果某个粒子适应度高于历史上的最佳适应度（pbest）
		将该值设置为新的pbest
	选择所有粒子的最佳适配值的粒子作为gbest
	for each_p in 粒子群:
		计算粒子速度
		更新粒子位置
'''

import numpy as np

class PSO():

    def __init__(self, pN, dim, max_iter, func, w=0.8, c1=2, c2=2, r1=0.6, r2=0.3):
        self.w = w                                                      # weight
        self.c1 = c1                                                    # self cognitive factor
        self.c2 = c2                                                    # social cognitive factor
        self.r1 = r1                                                    # self cognitive learning rate
        self.r2 = r2                                                    # social cognitive learning rate
        self.pN = pN                                                    # the number of particle
        self.dim = dim                                                  # Search dimension
        self.max_iter = max_iter                                        # Maximum iteration
        self.Pos = np.zeros((self.pN, self.dim))                        # The initial position of the particle
        self.V = np.zeros((self.pN, self.dim))                          # The initial velocity of the particle
        self.pbest = np.zeros((self.pN, self.dim), dtype=np.float32)    # The historical best position of the particle
        self.gbest = np.zeros((self.pN, self.dim), dtype=np.float32)    # The historical best position of all particle
        self.p_BestFit = np.zeros(self.pN)                              # The historical best fit for each particle
        self.fit = -1e15                                                # Global best fit
        self.func= func                                                 # function

    def function(self, x):
        return self.func(x)

    def initial_particle(self):                                         # Initialization of particle swarm
        for i in range(self.pN):
            self.Pos[i] = np.random.uniform(0, 5, (1, self.dim))
            self.V[i] = np.random.uniform(0, 5, (1, self.dim))

            self.pbest[i] = self.Pos[i]
            self.p_BestFit[i] = self.function(self.Pos[i])

            if self.p_BestFit[i] > self.fit:
                self.fit = self.p_BestFit[i]
                self.gbest = self.Pos[i]

    def update(self):
        fitness = []


        for _ in range(self.max_iter):
            for i in range(self.pN):  # update weight
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.Pos[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.Pos[i])
                self.Pos[i] = self.Pos[i] + self.V[i]

            fitness.append(self.fit)

            for i in range(self.pN):    # update gbest and pbest
                temp = self.function(self.Pos[i])
                if temp > self.p_BestFit[i]:    # update each particle
                    self.p_BestFit[i] = temp
                    self.pbest[i] = self.Pos[i]
                    if self.p_BestFit[i] > self.fit:
                        self.gbest = self.Pos[i]
                        self.fit = self.p_BestFit[i]



        return self.gbest, self.fit

# def count_func(x):
#     y = -x**2 + 20 * x + 10
#     return y
#
# pso_example = PSO(pN=5, dim=1, max_iter=300, func=count_func)
# pso_example.initial_particle()
# x_best, fit_best = pso_example.update()
# print(x_best, fit_best)