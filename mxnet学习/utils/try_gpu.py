def try_gpu():
    import mxnet
    from mxnet import nd
    try:
        nd.zeros((1,), ctx=mxnet.gpu())
    except mxnet.base.MXNetError:
        return mxnet.cpu()