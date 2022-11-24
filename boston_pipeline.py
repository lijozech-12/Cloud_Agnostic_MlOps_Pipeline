import kfp
from kfp import dsl

def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='lijozech12/boston_pipeline_images:preprocess',
        arguments=[],
        file_outputs={
            'x_train': '/app/x_train.npy',
            'x_test': '/app/x_test.npy',
            'y_train': '/app/y_train.npy',
            'y_test': '/app/y_test.npy',
        }
    )

def train_op(x_train, y_train):

    return dsl.ContainerOp(
        name='Train Model',
        image='lijozech12/boston_pipeline_images:train',
        arguments=[
            '--x_train', x_train,
            '--y_train', y_train
        ],
        file_outputs={
            'model': '/app/model.pkl'
        }
    )

def test_op(x_test, y_test, model):

    return dsl.ContainerOp(
        name='Test Model',
        image='lijozech12/boston_pipeline_images:test',
        arguments=[
            '--x_test', x_test,
            '--y_test', y_test,
            '--model', model
        ],
        file_outputs={
            'mean_squared_error': '/app/output.txt'
        }
    )

def deploy_model_op(model):

    return dsl.ContainerOp(
        name='Deploy Model',
        image='lijozech12/boston_pipeline_images:deploy_model',
        arguments=[
            '--model', model
        ]
    )

@dsl.pipeline(
   name='Boston Housing Pipeline',
   description='An example pipeline that trains and logs a regression model.'
)
def boston_pipeline():
    _preprocess_op = preprocess_op()
    
    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_train'])
    ).after(_preprocess_op)

    _test_op = test_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_test']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_test']),
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_train_op)

    deploy_model_op(
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_test_op)

kfp.compiler.Compiler().compile(
    pipeline_func=boston_pipeline,
    package_path='boston_pipeline.yaml')