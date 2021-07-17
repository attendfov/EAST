import tensorflow as tf

# common
tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/east_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_integer('input_size', 512, '')

# multigpu train
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 14, '')
tf.app.flags.DEFINE_integer('num_readers', 16, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')

# data layer
tf.app.flags.DEFINE_string('training_data_dir', '/data/ocr/icdar2015/',
                           'training dataset to use')
tf.app.flags.DEFINE_string('train_img_list_path', '',
                           'train image list path')
tf.app.flags.DEFINE_string('train_xml_list_path', '',
                           'train xml list path')
tf.app.flags.DEFINE_integer('max_image_large_side', 1280,
                            'max image size of training')
tf.app.flags.DEFINE_integer('max_text_size', 800,
                            'if the text in the input image is bigger than this, then we resize'
                            'the image according to this')
tf.app.flags.DEFINE_integer('min_text_size', 10,
                            'if the text size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.1,
                          'when doing random crop from input image, the'
                          'min length of min(H, W')
tf.app.flags.DEFINE_string('geometry', 'RBOX',
                           'which geometry to generate, RBOX or QUAD')
tf.app.flags.DEFINE_boolean('rotate_4dir', False, 'randomly rotate image in four directions')
tf.app.flags.DEFINE_boolean('ignore_large_rotate_poly', True, 'ignore text poly with large rotation')
tf.app.flags.DEFINE_bool('permute_points', True, 'permute points')

# infer
tf.app.flags.DEFINE_string('test_data_dir', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_integer('max_side_len', 512, 'max side length')
tf.app.flags.DEFINE_bool('save_score_map', False, 'if save score map or not')
tf.app.flags.DEFINE_bool('enable_box_refinement', False, 'refine boxes')
tf.app.flags.DEFINE_bool('save_infer_graph', True, 'save inference graph')


FLAGS = tf.app.flags.FLAGS