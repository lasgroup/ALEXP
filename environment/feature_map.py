import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from functools import cached_property
from typing import List, Optional, Union
import itertools
import math

class FeatureMap:

    def __init__(self, num_dim_x: int, random_state=None):
        assert num_dim_x > 0, f'num_dim_x must be a positive integer, but got {num_dim_x}'
        self._num_dim_x = num_dim_x
        self._rds = np.random if random_state is None else random_state

    @property
    def groups(self) -> List[List[int]]:
        raise NotImplementedError

    @cached_property
    def num_groups(self) -> int:
        return len(self.groups)

    @cached_property
    def size(self) -> int:
        return sum([len(g) for g in self.groups])

    @property
    def num_dim_x(self) -> int:
        return self._num_dim_x

    def get_feature(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PeriodicMap(FeatureMap):
    '''
    Returns a Cosine Map.
    '''
    def __init__(self, num_dim_x: int, num_groups: int, num_freqs_per_group: int = 1, base_freq: float = 2e-1 * np.pi):
        super().__init__(num_dim_x=num_dim_x)
        assert num_dim_x in [1, 2]
        self.num_freqs_per_group = num_freqs_per_group
        self.base_freq = base_freq

        # group frequencies of the same range (e.g. (0, 1, 2) and (3, 4, 5))
        freqs = base_freq * np.array([i * num_freqs_per_group + j for i in range(num_groups)
                                      for j in range(num_freqs_per_group)])
        if num_dim_x == 1:
            self._freqs = np.expand_dims(freqs, axis=-1)
        elif num_dim_x == 2:
            self._freqs = np.transpose([np.tile(freqs, len(freqs)), np.repeat(freqs, len(freqs))])
        else:
            raise NotImplementedError('Only num_dim_x = 1, 2 supported.')
        assert self._freqs.shape[-1] == num_dim_x
        self._groups = np.split(np.arange(2 * (num_groups * num_freqs_per_group)**num_dim_x), num_groups**num_dim_x)

    def get_feature(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 0:
            x = np.expand_dims(x, axis=-1)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)
        assert x.ndim == 2 and x.shape[-1] == self._freqs.shape[-1]
        freq_x = np.matmul(x, self._freqs.T)
        features = np.stack([np.sin(freq_x), np.cos(freq_x)], axis=-1)
        features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2]), order='C')
        assert features.shape[0] == x.shape[0] and features.shape[-1] == self.size
        return features

    @property
    def groups(self) -> List[List[int]]:
        return self._groups


class LinearMap(FeatureMap):

    def __init__(self, num_dim_x: int, one_group_per_dim: bool = False):
        super().__init__(num_dim_x)
        self.one_group_per_dim = one_group_per_dim

    @cached_property
    def groups(self) -> List[Union[List[int], np.ndarray]]:
        if self.one_group_per_dim:
            return [np.array([i]) for i in range(self.num_dim_x)]
        else:
            return [np.arange(self.num_dim_x)]

    def get_feature(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)
        assert x.ndim == 2 and x.shape[-1] == self.num_dim_x
        return x


class RFFMap(FeatureMap):

    def __init__(self, num_dim_x: int, feature_dim: int, lengthscale: float = 1.0, random_state=None):
        super(RFFMap, self).__init__(num_dim_x, random_state)
        self.feature_dim = feature_dim
        self.lengthscale = lengthscale
        self._w  = self._rds.normal(scale=1 / lengthscale, size=(feature_dim, num_dim_x))
        self._b = self._rds.uniform(0, 2*np.pi, size=(feature_dim,))

    @cached_property
    def groups(self) -> List[List[int]]: #does not depend on input dimension
        return [[i] for i in range(self.feature_dim)]

    def get_feature(self, x: np.ndarray) -> np.ndarray:
        # for now we sample w from a gaussian with diagonal * variance covariance matrix
        if x.ndim == 0:
            assert self.num_dim_x == 1
            x = x.reshape((1, 1))
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)
        assert x.ndim == 2, x.shape[-1] == self.num_dim_x
        rff = np.sqrt(2) * np.sin(np.dot(x, self._w.T) + self._b)
        assert rff.shape == (x.shape[0], self.feature_dim)
        return rff


class PolynomialMap(FeatureMap):

    def __init__(self, num_dim_x: int, degree: int = 6, one_group_per_dim: bool = False):
        super().__init__(num_dim_x)
        self.degree = degree
        self.one_group_per_dim = one_group_per_dim
        # if origin is None:
        #     self._origin = np.random.uniform(-0.99, 0.99, degree)
        # else:
        #     self._origin = origin

    @cached_property
    def groups(self) -> List[List[int]]:
        if self.one_group_per_dim:
            size = self.get_feature(2 * np.ones((1, self.num_dim_x))).shape[-1]
            return [[i] for i in range(size)]
        else:
            # group together features of same order
            phi = self.get_feature(2 * np.ones((1, self.num_dim_x)))[0]
            last_value = 1.
            groups = []
            current_group = []
            for i in range(phi.shape[-1]):
                if phi[i] > last_value:
                    groups.append(current_group)
                    current_group = [i]  # start new group
                    last_value = phi[i]
                else:
                    current_group += [i]
            groups.append(current_group)
        return groups

    def get_feature(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 0:
            assert self.num_dim_x == 1
            x = x.reshape((1, 1))
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)
        assert x.ndim == 2 and x.shape[-1] == self.num_dim_x
        poly = PolynomialFeatures(degree=self.degree)
        features = poly.fit_transform(x)
        return features


class LegendreMap(FeatureMap):

    def __init__(self, num_dim_x: int,
                 degree: int = 5,
                 domain_l: Optional[np.ndarray] = None,
                 domain_u: Optional[np.ndarray] = None):
        super().__init__(num_dim_x)
        assert num_dim_x in [1, 2], f'num_dim_x must be 1 or 2, but got {num_dim_x}'
        self.degree = degree

        # check and set legendre polynomial domain
        assert domain_l is None or domain_l.shape == (self.num_dim_x, )
        assert domain_u is None or domain_u.shape == (self.num_dim_x,)
        self._domain_l = domain_l if domain_l is not None else - 1.1 * np.ones(num_dim_x)
        self._domain_u = domain_u if domain_u is not None else 1.1 * np.ones(num_dim_x)

    @cached_property
    def groups(self) -> List[List[int]]:
        if self.num_dim_x == 1:
            group_inds = [[i] for i in range(self.degree+1)]
        elif self.num_dim_x == 2:
            assert self.num_dim_x == 2
            group_inds = [[i] for i in range(self._num_features)] #one term per group
        else:
            raise NotImplementedError
        # group_inds = [[] for i in range((self.degree + 1))] #one degree per group: this didn't give good result
        # indices = np.arange((self.degree+1)**2)
        # indices = indices.reshape((self.degree+1, self.degree+1))
        # for i, j in np.ndindex(indices.shape):
        #     group_num = np.max((i,j))
        #     group_inds[group_num].append(indices[i,j])
        return group_inds

    def get_feature(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 0:
            assert self.num_dim_x == 1
            x = x.reshape((1, 1))
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)
        assert x.ndim == 2, x.shape[-1] == self.num_dim_x
        if self.num_dim_x == 1:
            features = np.zeros((x.shape[0], self.degree+1))
            for i in range(self.degree+1):
                features[:, i] = np.squeeze(self._legendre_feature_fn(x, i+1, dim_of_x=0))
            return features
        elif self.num_dim_x == 2:
            features = np.zeros((x.shape[0], self._num_features))
            feature_id = 0
            for i in range(self.degree+1):
                for j in range(self.degree+1):
                    if i+j <= self.degree:
                        features[:, feature_id] = (np.squeeze(self._legendre_feature_fn(x[:, 0], i, dim_of_x=0)) *
                                                   np.squeeze(self._legendre_feature_fn(x[:, 1], j, dim_of_x=1)))
                        feature_id += 1
            assert feature_id == features.shape[-1]
            return features
        else:
            raise NotImplementedError('Can only support 1 and 2-dimensional x')

    def _legendre_feature_fn(self, x: np.ndarray, degree: int, dim_of_x: int):
        assert dim_of_x < self.num_dim_x
        assert degree >= 0
        coef = np.zeros(degree + 1)
        coef[-1] = 1
        poly = np.polynomial.Legendre(coef, domain=[self._domain_l[dim_of_x], self._domain_u[dim_of_x]])
        return poly(x)

    @property
    def _num_features(self) -> int:
        p = self.degree + 1
        if self.num_dim_x == 1:
            return p
        elif self.num_dim_x == 2:
            return p * (p + 1) // 2
        else:
            raise NotImplementedError


class CombOfLegendreMaps(FeatureMap):
    def __init__(self, num_dim_x: int,
                 max_degree: int = 10,
                 sparsity: int = 4,
                 domain_l: Optional[np.ndarray] = None,
                 domain_u: Optional[np.ndarray] = None):
        super().__init__(num_dim_x)
        assert num_dim_x in [1], f'num_dim_x must be 1, but got {num_dim_x}'
        self.max_degree = max_degree
        self.sparsity = sparsity

        # built the groups:
        combinations = list(itertools.combinations(range(0, self.max_degree + 1), self.sparsity))
        # Convert each combination tuple to a list
        self.combinations = [list(combination) for combination in combinations]
        # double check
        assert len(self.combinations) == math.comb(self.max_degree+1, self.sparsity)
        self.length = len(self.combinations)*self.sparsity

        # check and set legendre polynomial domain
        assert domain_l is None or domain_l.shape == (self.num_dim_x, )
        assert domain_u is None or domain_u.shape == (self.num_dim_x,)
        self._domain_l = domain_l if domain_l is not None else - 1.1 * np.ones(num_dim_x)
        self._domain_u = domain_u if domain_u is not None else 1.1 * np.ones(num_dim_x)

    @cached_property
    def groups(self) -> List[List[int]]:
        if self.num_dim_x == 1:
            arr = np.arange(self.length)
            # Reshape the array into a 2D array with rows of length a
            arr = arr.reshape((-1, self.sparsity))
            # Convert the 2D array into a list of lists
            group_inds = arr.tolist()
        else:
            raise NotImplementedError
        return group_inds

    def get_feature(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 0:
            assert self.num_dim_x == 1
            x = x.reshape((1, 1))
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)
        assert x.ndim == 2, x.shape[-1] == self.num_dim_x
        if self.num_dim_x == 1:
            features = np.zeros((x.shape[0], self.length))
            ind_count = 0
            for ind_list in self.combinations:
                for i in ind_list:
                    features[:, ind_count] = np.squeeze(self._legendre_feature_fn(x, i, dim_of_x=0))
                    ind_count += 1
            return features
        else:
            raise NotImplementedError('Can only support 1-dimensional x')

    def _legendre_feature_fn(self, x: np.ndarray, degree: int, dim_of_x: int):
        assert dim_of_x < self.num_dim_x
        assert degree >= 0
        coef = np.zeros(degree + 1)
        coef[-1] = 1
        poly = np.polynomial.Legendre(coef, domain=[self._domain_l[dim_of_x], self._domain_u[dim_of_x]])
        return poly(x)

    @property
    def _num_features(self) -> int:
        if self.num_dim_x == 1:
            return self.length
        else:
            raise NotImplementedError


class FilteredFeatureMap(FeatureMap):

    def __init__(self, feature_map: FeatureMap, eta: np.ndarray):
        super().__init__(num_dim_x=feature_map.num_dim_x)

        assert eta.shape == (feature_map.num_groups,)
        self.eta = eta
        self.active_groups = np.where(eta > 0.0)[0]
        self._wrapped_feature_map = feature_map

    @cached_property
    def groups(self) -> List[List[int]]:
        return [g for i, g in enumerate(self._wrapped_feature_map.groups) if i in self.active_groups]

    @property
    def num_dim_x(self) -> int:
        return self._wrapped_feature_map.num_dim_x

    @property
    def active_indices(self) -> np.ndarray:
        return np.concatenate(self.groups)

    def get_feature(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)
        assert x.ndim == 2 and x.shape[-1] == self.num_dim_x
        filtered_features = self._wrapped_feature_map.get_feature(x)[:, self.active_indices]
        #filtered_features *= np.sqrt(self.eta[self.active_indices])
        assert filtered_features.shape == (x.shape[0], self.size)
        return filtered_features


class UnionOfFeatureMaps(FeatureMap):
    def __init__(self, feature_maps: List[FeatureMap]):
        assert len(set([feature_map.num_dim_x for feature_map in feature_maps])) == 1, (
            'The feature_maps must all have the same num_dim_x')
        super().__init__(num_dim_x=feature_maps[0].num_dim_x)

        self._feature_maps = feature_maps

        # setup groups
        _current_size = 0
        self._groups = []
        for feature_map in feature_maps:
            _current_size += feature_map.size
        self._groups = [[i] for i in range(_current_size)]
        # print('from groups:', self.size)
        # print(_current_size)
        assert _current_size == self.size
        # print(self.size)
        # assert set().union(*self.groups) == set(range(self.size))

    @property
    def groups(self) -> List[List[int]]:
        return self._groups

    def get_feature(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)
        assert x.ndim == 2 and x.shape[-1] == self.num_dim_x
        feature = np.concatenate([feature_map.get_feature(x) for feature_map in self._feature_maps], axis=-1)
        assert feature.shape == (x.shape[0], self.size)
        return feature

class ProductOfMaps(FeatureMap):
    def __init__(self, map1: FeatureMap, map2: FeatureMap):
        assert len(set([feature_map.num_dim_x for feature_map in [map1, map2]])) == 1, (
            'The feature_maps must all have the same num_dim_x')
        super().__init__(num_dim_x=map1.num_dim_x)
        self.map1 = map1
        self.map2 = map2
        cross_size = map1.size * map2.size
        self._groups = [[i] for i in range(cross_size)]

    @property
    def groups(self) -> List[List[int]]:
        return self._groups

    def get_feature(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)
        assert x.ndim == 2 and x.shape[-1] == self.num_dim_x
        feat1 = self.map1.get_feature(x).squeeze()
        feat2 = self.map2.get_feature(x).squeeze()
        prod_feat = []
        for row1, row2 in zip(feat1, feat2):
            prod_feat.append(np.tensordot(row1.squeeze(), row2.squeeze(), axes = [[],[]]).flatten())
        return np.array(prod_feat)




if __name__ == '__main__':
    x = np.ones(5)
    map = CombOfLegendreMaps(num_dim_x=1, sparsity=2, max_degree=3)
    print(len(map.groups))
    phi = map.get_feature(x)
    print('CombOfLeg:', phi)

