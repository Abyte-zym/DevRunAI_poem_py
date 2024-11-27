# 下载代码和数据集
import os
import moxing as mox
mox.file.copy_parallel(
    'obs://obs-aigallery-zc/hyx/tensorflow_poems-master/', './tensorflow_poems-master/')
# %cd ./tensorflow_poems-master/

# 环境导入

# 参数设置
parser = argparse.ArgumentParser()
# type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('--batch_size', type=int, help='batch_size', default=64)
parser.add_argument('--learning_rate', type=float,
                    help='learning_rate', default=0.0001)
parser.add_argument('--model_dir', type=Path,
                    help='model save path.', default='./model')
parser.add_argument('--file_path', type=Path,
                    help='file name of poems.', default='./data/poems.txt')
parser.add_argument('--model_prefix', type=str,
                    help='model save prefix.', default='poems')
parser.add_argument('--epochs', type=int,
                    help='train how many epochs.', default=126)

args = parser.parse_args(args=[])

# 训练


def run_training():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    poems_vector, word_to_int, vocabularies = process_poems(args.file_path)
    batches_inputs, batches_outputs = generate_batch(
        args.batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [args.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [args.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=args.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(args.model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('## start training...')
        try:
            n_chunk = len(poems_vector) // args.batch_size
            for epoch in range(start_epoch, args.epochs):
                n = 0
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                print('Epoch: %d, batch: %d, training loss: %.6f' %
                      (epoch, batch, loss))
                # if epoch % 5 == 0:
                saver.save(sess, os.path.join(args.model_dir,
                           args.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(args.model_dir,
                       args.model_prefix), global_step=epoch)
            print(
                '## Last epoch were saved, next time will start from epoch {}.'.format(epoch))


run_training()

# 诗词生成

start_token = 'B'
end_token = 'E'
model_dir = './model/'
corpus_file = './data/poems.txt'

lr = 0.0002


def to_word(predict, vocabs):
    predict = predict[0]
    predict /= np.sum(predict)
    sample = np.random.choice(np.arange(len(predict)), p=predict)
    if sample > len(vocabs):
        return vocabs[-1]
    else:
        return vocabs[sample]


def gen_poem(begin_word):
    batch_size = 1
    print('## loading corpus from %s' % model_dir)
    tf.reset_default_graph()

    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=lr)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        word = begin_word or to_word(predict, vocabularies)
        poem_ = ''

        i = 0
        while word != end_token:
            poem_ += word
            i += 1
            if i > 24:
                break
            x = np.array([[word_int_map[word]]])
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)

        return poem_


def pretty_print_poem(poem_):
    poem_sentences = poem_.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


if __name__ == '__main__':
    begin_char = input(
        '## （输入 quit 退出）请输入第一个字 please input the first character: ')
    if begin_char == 'quit':
        exit()
    poem = gen_poem(begin_char)
    pretty_print_poem(poem_=poem)
