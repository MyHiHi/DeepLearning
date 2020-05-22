def evaluate_acc(data_iter, net, ctx):
    from mxnet import nd
    acc_sum, n = nd.array([0], ctx=ctx), len(data_iter)
    for X, y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).mean()
    return acc_sum.asscalar()/n
