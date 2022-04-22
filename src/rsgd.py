import torch.optim
from manifolds import Manifold, Euclidean, ManifoldParameter

# in order not to create it at each iteration
_default_manifold = Euclidean()

import re


def insert_docs(doc, pattern=None, repl=None):
    def wrapper(fn):
        # assume wrapping
        if pattern is not None:
            if repl is None:
                raise RuntimeError("need repl parameter")
            fn.__doc__ = re.sub(pattern, repl, doc)
        else:
            fn.__doc__ = doc
        return fn

    return wrapper
class ManifoldTensor(torch.Tensor):
    """Same as :class:`torch.Tensor` that has information about its manifold.
    Other Parameters
    ----------------
    manifold : :class:`geoopt.Manifold`
        A manifold for the tensor, (default: :class:`geoopt.Euclidean`)
    """

    try:
        # https://github.com/pytorch/pytorch/issues/46159#issuecomment-707817037
        from torch._C import _disabled_torch_function_impl  # noqa

        __torch_function__ = _disabled_torch_function_impl

    except ImportError:
        pass

    def __new__(
        cls, *args, manifold: Manifold = Euclidean(), requires_grad=False, **kwargs
    ):
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            data = args[0].data
        else:
            data = torch.Tensor(*args, **kwargs)
        if kwargs.get("device") is not None:
            data.data = data.data.to(kwargs.get("device"))
        with torch.no_grad():
            manifold.assert_check_point(data)
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.manifold = manifold
        return instance

    manifold: Manifold

    def proj_(self) -> torch.Tensor:
        """
        Inplace projection to the manifold.
        Returns
        -------
        tensor
            same instance
        """
        return self.copy_(self.manifold.projx(self))

    @insert_docs(Manifold.retr.__doc__, r"\s+x : .+\n.+", "")
    def retr(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.retr(self, u=u, **kwargs)

    @insert_docs(Manifold.expmap.__doc__, r"\s+x : .+\n.+", "")
    def expmap(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.expmap(self, u=u, **kwargs)

    @insert_docs(Manifold.inner.__doc__, r"\s+x : .+\n.+", "")
    def inner(self, u: torch.Tensor, v: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.manifold.inner(self, u=u, v=v, **kwargs)

    @insert_docs(Manifold.proju.__doc__, r"\s+x : .+\n.+", "")
    def proju(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.proju(self, u, **kwargs)

    @insert_docs(Manifold.transp.__doc__, r"\s+x : .+\n.+", "")
    def transp(self, y: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.transp(self, y, v, **kwargs)

    @insert_docs(Manifold.retr_transp.__doc__, r"\s+x : .+\n.+", "")
    def retr_transp(
        self, u: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.manifold.retr_transp(self, u, v, **kwargs)

    @insert_docs(Manifold.expmap_transp.__doc__, r"\s+x : .+\n.+", "")
    def expmap_transp(self, u: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.expmap_transp(self, u, v, **kwargs)

    @insert_docs(Manifold.transp_follow_expmap.__doc__, r"\s+x : .+\n.+", "")
    def transp_follow_expmap(
        self, u: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.manifold.transp_follow_expmap(self, u, v, **kwargs)

    @insert_docs(Manifold.transp_follow_retr.__doc__, r"\s+x : .+\n.+", "")
    def transp_follow_retr(
        self, u: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.manifold.transp_follow_retr(self, u, v, **kwargs)

    def dist(
        self, other: torch.Tensor, p: Union[int, float, bool, str] = 2, **kwargs
    ) -> torch.Tensor:
        """
        Return euclidean  or geodesic distance between points on the manifold. Allows broadcasting.
        Parameters
        ----------
        other : tensor
        p : str|int
            The norm to use. The default behaviour is not changed and is just euclidean distance.
            To compute geodesic distance, :attr:`p` should be set to ``"g"``
        Returns
        -------
        scalar
        """
        if p == "g":
            return self.manifold.dist(self, other, **kwargs)
        else:
            return super().dist(other)

    @insert_docs(Manifold.logmap.__doc__, r"\s+x : .+\n.+", "")
    def logmap(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.logmap(self, y, **kwargs)

    def __repr__(self):
        return "Tensor on {} containing:\n".format(
            self.manifold
        ) + torch.Tensor.__repr__(self)

    # noinspection PyUnresolvedReferences
    def __reduce_ex__(self, proto):
        build, proto = super(ManifoldTensor, self).__reduce_ex__(proto)
        new_build = functools.partial(_rebuild_manifold_tensor, build_fn=build)
        new_proto = proto + (dict(), self.__class__, self.manifold, self.requires_grad)
        return new_build, new_proto

    @insert_docs(Manifold.unpack_tensor.__doc__, r"\s+tensor : .+\n.+", "")
    def unpack_tensor(self) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        return self.manifold.unpack_tensor(self)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format),
                manifold=copy.deepcopy(self.manifold, memo=memo),
                requires_grad=self.requires_grad,
            )
            memo[id(self)] = result
            return result
class OptimMixin(object):
    def __init__(self, *args, stabilize=None, **kwargs):
        self._stabilize = stabilize
        super().__init__(*args, **kwargs)

    def stabilize_group(self, group):
        pass

    def stabilize(self):
        """Stabilize parameters if they are off-manifold due to numerical reasons
        """
        for group in self.param_groups:
            self.stabilize_group(group)


def copy_or_set_(dest, source):
    """
    A workaround to respect strides of :code:`dest` when copying :code:`source`
    (https://github.com/geoopt/geoopt/issues/70)
    Parameters
    ----------
    dest : torch.Tensor
        Destination tensor where to store new data
    source : torch.Tensor
        Source data to put in the new tensor
    Returns
    -------
    dest
        torch.Tensor, modified inplace
    """
    if dest.stride() != source.stride():
        return dest.copy_(source)
    else:
        return dest.set_(source)


class RiemannianSGD(OptimMixin, torch.optim.Optimizer):
    r"""
    Riemannian Stochastic Gradient Descent with the same API as :class:`torch.optim.SGD`.
    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float
        learning rate
    momentum : float (optional)
        momentum factor (default: 0)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    dampening : float (optional)
        dampening for momentum (default: 0)
    nesterov : bool (optional)
        enables Nesterov momentum (default: False)
    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        stabilize=None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults, stabilize=stabilize)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                learning_rate = group["lr"]
                group["step"] += 1
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianSGD does not support sparse gradients, use SparseRiemannianSGD instead"
                        )
                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        if momentum > 0:
                            state["momentum_buffer"] = grad.clone()
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        manifold = point.manifold
                    else:
                        manifold = self._default_manifold

                    grad.add_(point, alpha=weight_decay)
                    grad = manifold.egrad2rgrad(point, grad)
                    if momentum > 0:
                        momentum_buffer = state["momentum_buffer"]
                        momentum_buffer.mul_(momentum).add_(grad, alpha=1 - dampening)
                        if nesterov:
                            grad = grad.add_(momentum_buffer, alpha=momentum)
                        else:
                            grad = momentum_buffer
                        # we have all the things projected
                        new_point, new_momentum_buffer = manifold.retr_transp(
                            point, -learning_rate * grad, momentum_buffer
                        )
                        momentum_buffer.copy_(new_momentum_buffer)
                        # use copy only for user facing point
                        point.copy_(new_point)
                    else:
                        new_point = manifold.retr(point, -learning_rate * grad)
                        point.copy_(new_point)

                if (
                    group["stabilize"] is not None
                    and group["step"] % group["stabilize"] == 0
                ):
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            manifold = p.manifold
            momentum = group["momentum"]
            p.copy_(manifold.projx(p))
            if momentum > 0:
                param_state = self.state[p]
                if not param_state:  # due to None grads
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.copy_(manifold.proju(p, buf))
