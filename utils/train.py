def train(train_iter, test_iter, batch_size, net, epochs, lr, ctx,trainer, loss, evaluate_acc):
    import time
    
    from mxnet import autograd
    
    for e in range(epochs):
        train_loss, train_acc, test_acc, start, n = .0, .0, .0, time.time(), len(train_iter)
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                out = net(X)
                l = loss(out, y)
            l.backward()
            trainer.step(batch_size)
            train_loss += l.mean().asscalar()
            y = y.astype('float32')
            train_acc += (out.argmax(axis=1) == y).mean().asscalar()
        test_acc = evaluate_acc(test_iter, net, ctx)
        print('epoechs:%d, loss: %.4f, train_acc:%.3f, test_acc:%.3f ,used time: %.1f sec' % (
            e+1, train_loss/n, train_acc/n, test_acc, time.time()-start))
