# Basic
import csv
import pandas as pd
import numpy as np
import random as rd
import copy
import os
import sklearn
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action = 'ignore')

from pathlib import Path
import collections
import sys
import matplotlib.pyplot as plt

# Sklearn Metrics
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
sys.path.insert(1, sys.path[0] + '/Preprocessing')

# Preprocessing & Models 
from sklearn.preprocessing import MinMaxScaler

def uniform_sampling(features_range) : 
    current = {}
    for k in features_range.keys() : 
        current[k] = round(rd.uniform(features_range[k][0], features_range[k][1]), 4)
    return current

def merge_sort(sort_target) : 
    if len(sort_target) <= 1 : 
        return sort_target
    mid = len(sort_target) // 2
    left = sort_target[:mid]
    right = sort_target[mid:]
    left = merge_sort(left)
    right = merge_sort(right)
    return merge(left, right)

def merge(left, right) : 
    result = []
    while len(left) > 0 or len(right) > 0 : 
        if len(left) > 0 and len(right) > 0 : 
            if left[0].value <= right[0].value : 
                result.append(left[0])
                left = left[1:]
            else : 
                result.append(right[0])
                right = right[1:]
        elif len(left) > 0 : 
            result.append(left[0])
            left = left[1:]
        elif len(right) > 0 : 
            result.append(right[0])
            right = right[1:]
    return result

class unit : 
    def __init__(self, obj, features, additional_info) : 
        """
        This class represent a possible solution of GA 
        
        Parameters 
        - obj : Function / See "obj" parameter of "GeneticAlgorithm" class 
        - features : List / the solution (features) of this unit
        - additional_info : Dict / See "addtional_info" parameter of "GeneticAlgorithm" class 
        
        Attributes 
        - features : List / the solution (features) of this unit (same with the parameter "features")
        - value : The calculated value of given objective function 
        """
        self.features = features
        self.value = obj(features, additional_info)

class iteration :  
    def __init__(self, mating, obj, opt_to, additional_info) : 
        """
        This class represent a iteration while running GA
        
        Parameters 
        - obj : Function / See "obj" parameter of "GeneticAlgorithm" class    
        - opt_to : Str / See "optimize_to" parameter of "GeneticAlgorithm" class
        
        
        Attributes 
        - features : List / the solution (features) of this unit (same with the parameter "features")
        - value : The calculated value of given objective function 
        """            
        self.mating = mating
        self.obj = obj
        self.opt_to = opt_to
        self.additional_info = additional_info
        
        self.key_list = None
        self.units = []
        
    def duplicate_check(self, features, units) :
        same = False 
        for u in units : 
            if features == u.features : 
                same = True
        return same
    
    def initial_iteration(self, n_pop, features_range) : 
        self.key_list = features_range.keys()
        self.units.append(unit(self.obj, uniform_sampling(features_range), self.additional_info))
        for n in range(1, n_pop) : 
            current = uniform_sampling(features_range)
            same = self.duplicate_check(current, self.units)
            while same : 
                current = uniform_sampling(features_range)     
                same = self.duplicate_check(current, self.units)
            self.units.append(unit(self.obj, current, self.additional_info))
        
        self.units = merge_sort(self.units)
        if self.opt_to == 'maximize' : 
            self.units.reverse()
    
    def elite_mating(self, n_elite, elites, features_range, units) : 
        for i in range(0, n_elite) : 
            parent1 = elites[i]
            for j in range(i, n_elite) :
                parent2 = elites[j]
                current = self.mating(parent1, parent2, self.key_list)
                if self.duplicate_check(current, units) : 
                    current_unit = unit(self.obj, uniform_sampling(features_range), self.additional_info)
                else : 
                    current_unit = unit(self.obj, current, self.additional_info)
                units.append(current_unit)
        return units
    
    def add_mutation(self, mutation, features_range, units) : 
        for m in range(0, mutation) : 
            units.append(unit(self.obj, uniform_sampling(features_range), self.additional_info))
        return units 
    
    def not_elite_mating(self, number_to_mate, n_pop, n_elite, features_range, units) : 
        for i in range(0, number_to_mate) : 
            parent1 = rd.randrange(n_elite, n_pop)
            parent2 = parent1
            while parent1 == parent2 : 
                parent2 = rd.randrange(n_elite, n_pop)
            parent1 = self.units[parent1]
            parent2 = self.units[parent2]
            current = self.mating(parent1, parent2, self.key_list)
            if self.duplicate_check(current, units) : 
                current_unit = unit(self.obj, uniform_sampling(features_range), self.additional_info)
            else : 
                current_unit = unit(self.obj, current, self.additional_info)
            units.append(current_unit)
        return units     
    
    def iter_run(self, n_pop, n_elite, mutation, features_range) : 
        """
        The function to generate units from previous generation
        Describe the logic of this implementation 
        If you have better or different idea to make the next generation, describe your logic and implement it. (Optional, Extra points)
                
        Parameters
        - n_pop : Int / See "n_population" parameter of "GeneticAlgorithm" class   
        - n_elite : Int / See "n_elite" parameter of "GeneticAlgorithm" class  
        - mutation : Int / See "mutation" parameter of "GeneticAlgorithm" class  
        - features_range : Dict of Lists / See "features_range" parameter of "GeneticAlgorithm" class 
        
        Used functions 
        - self.elite_mating : Make child between parents that are elites
        - self.add_mutation : Make mutation child
        - self.not_elite_mating : Make child between the parents that are not elites
        """
        elites = self.units[:n_elite] 
        units = copy.deepcopy(elites) # the elites of previous iteration remain 
        
        units = self.elite_mating(n_elite, elites, features_range, units) 
        
        if len(units) > n_pop-mutation : 
            units = merge_sort(units)
            if self.opt_to == 'maximize' : 
                units.reverse()
            units = units[:n_pop-mutation]        
        
        units = self.add_mutation(mutation, features_range, units)
        
        number_to_generate = n_pop - len(units)
        units = self.not_elite_mating(number_to_generate, n_pop, n_elite, features_range, units)
        
        self.units = merge_sort(units)
            

class GeneticAlgorithm : 
    def __init__(
        self,  
        mating_function, 
        n_population: int = 10, 
        max_iteration: int = 100, 
        n_elite: int = 2, 
        mutation: int = 4, 
        optimize_to: str = 'minimize',
        early_stop: int = 0,
        printing: int = 10,
        ) : 
        """
        Genetic Algorithm 
        
        - Parameters 
        mating_function : Func / The function to make a child from two units
        n_population : Int / the number of units 
        max_iteration : Int / the terminal point
        n_elite : Int / the number of elites
        mutation : Int / the number of mutation, there will be no mutation if set as 0
        optimize_to : Str / 'minimize' or 'maximize'
        early_stop : Int / Stop before max_iteration when the optimal value do not change, there will be no early stop if set as 0
        printing : Int /  Print the value of the iteration during running, there will be no printing if set as 0
        
        - Attributes 
        iteration : Class "iteration" / initial and final iteration of GA at before and after running respectively
        best_unit : Class "unit" / The best unit after running 
        best_features : List / The features of solution after running 
        best_result : The optimal value of objective function after running
        """
        self.mating_function = mating_function
        self.n_pop = n_population
        self.max_iter = max_iteration
        self.n_elite = n_elite
        self.mutation = mutation
        self.opt_to = optimize_to
        if early_stop == 0 : 
            self.early_stop = max_iteration
        else : 
            self.early_stop = early_stop
        if printing == 0 : 
            self.printing = max_iteration
        else : 
            self.printing = printing
        
        self.iteration = None
        self.best_unit = None
        self.best_features = None
        self.best_result = None
    
    def run(self, obj, features_range, const = None, additional_info = None) : 
        """
        The function to run GA
        
        - Parameters 
        obj : Function / the objective function to optimize
        features_range : List of Lists / the min and max range of features
        const : List of Functions / the constraints of the problem (optional)
        additional_info : Dict / the additional information for optimization (e.g. prediction model)
        """
        self.iteration = iteration(self.mating_function, obj, self.opt_to, additional_info)
        self.iteration.initial_iteration(self.n_pop, features_range)
        
        early_stop_num = 0
        if self.opt_to == 'minimize' : 
            best = 1e+10000
        elif self.opt_to =='maximize' : 
            best = -1 * 1e+10000
        for i in range(0, self.max_iter) : 
            # You have to implement the function "iter_run" of class "iteration"
            self.iteration.iter_run(self.n_pop, self.n_elite, self.mutation, features_range)
            
            current_value = self.iteration.units[0].value # the best value of current iteration
            if current_value == best : 
                early_stop_num += 1
            best = current_value 
            
            if (i+1)%self.printing == 0 : 
                print(
                    "Iteration {} | Best {}".format(i+1, best)
                )
                
            if early_stop_num > self.early_stop : 
                break
        
        best_unit = self.iteration.units[0]
        self.best_features = best_unit.features
        self.best_result = best_unit.value
        
                
                
            
            
                        
                     
    
