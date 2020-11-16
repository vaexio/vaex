from .s3fs import translate_options

def test_translate():
    assert translate_options({
        'anonymous': False,
        'access_key': 'acces',
        'secret_key': 'secret'
    }) == {
        'anon': False,
        'client_kwargs': {
            'aws_access_key_id': 'acces',
            'aws_secret_access_key': 'secret'
        }
    }