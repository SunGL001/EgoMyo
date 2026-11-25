"""Deterministic cross-trial, cross-session and cross-subject splits.

The three protocols separate increasing amounts of the recording conditions.
Cross-trial holds out repetitions of every action, cross-session holds out the
entire second session, and cross-subject holds out whole participants with both
hands and both sessions. Every ranking uses a salted SHA-256 key, so the split
is reproducible from the membership file alone and does not depend on file
order or on the host.
"""
import hashlib
import json

PROTOCOLS = ('cross_trial', 'cross_session', 'cross_subject')
TRAIN_REPETITIONS = 3
TRAIN_SUBJECT_FRACTION = .7


def _rank(values, salt):
    return sorted(values, key=lambda value: (hashlib.sha256(f'{salt}:{value}'.encode()).hexdigest(),
                                             str(value)))


def _check(rows, split):
    train, test = set(split['train']), set(split['test'])
    if len(train) != len(split['train']) or len(test) != len(split['test']):
        raise ValueError('Duplicate trial id inside a split')
    if train & test:
        raise ValueError('Train and test overlap')
    if not train or not test:
        raise ValueError('Empty train or test split')
    if not (train | test) <= {row['trial_id'] for row in rows}:
        raise ValueError('Unknown trial id in split')


def build_splits(membership, split_seed=0):
    rows = membership['trials']
    by_id = {row['trial_id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate trial id in membership')
    participants = sorted({row['participant'] for row in rows})
    if len(participants) < 4:
        raise ValueError('Cross-subject protocol needs at least four participants')
    sessions = {row['session'] for row in rows}
    if sessions != {1, 2}:
        raise ValueError('Both capture sessions are required')

    strata = {}
    for row in rows:
        strata.setdefault((row['package'], row['session'], row['action']), []).append(row)
    for key, group in strata.items():
        if len(group) != 5 or {row['repetition'] for row in group} != {1, 2, 3, 4, 5}:
            raise ValueError(f'Incomplete repetition stratum {key}')

    trial_train = set()
    for key, group in sorted(strata.items()):
        order = _rank([row['trial_id'] for row in group], f'cross-trial:{split_seed}:{key}')
        trial_train.update(order[:TRAIN_REPETITIONS])

    subject_count = int(round(TRAIN_SUBJECT_FRACTION * len(participants)))
    subject_train = set(_rank(participants, f'cross-subject:{split_seed}')[:subject_count])
    subject_test = set(participants) - subject_train
    if subject_train & subject_test:
        raise ValueError('Participant leakage across the subject split')

    ids = set(by_id)
    splits = {
        'cross_trial': dict(train=sorted(trial_train), test=sorted(ids - trial_train)),
        'cross_session': dict(train=sorted(u for u, row in by_id.items() if row['session'] == 1),
                              test=sorted(u for u, row in by_id.items() if row['session'] == 2)),
        'cross_subject': dict(train=sorted(u for u, row in by_id.items()
                                           if row['participant'] in subject_train),
                              test=sorted(u for u, row in by_id.items()
                                          if row['participant'] in subject_test)),
    }
    for name, split in splits.items():
        _check(rows, split)
    return dict(
        schema='myobridge.splits.v1', split_seed=split_seed,
        protocols=splits,
        definitions=dict(
            cross_trial=dict(stratum=['package', 'session', 'action'],
                             repetitions_per_stratum=5,
                             training_repetitions=TRAIN_REPETITIONS,
                             testing_repetitions=5 - TRAIN_REPETITIONS,
                             ranking='salted SHA-256 over trial id'),
            cross_session=dict(training_session=1, testing_session=2,
                               direction='session 1 to session 2'),
            cross_subject=dict(unit='participant, both hands and all sessions',
                               requested_train_fraction=TRAIN_SUBJECT_FRACTION,
                               train_participants=sorted(subject_train),
                               test_participants=sorted(subject_test),
                               ranking='salted SHA-256 over participant id')),
        test_accessed=False)


def load_splits(path):
    with open(path) as stream:
        return json.load(stream)
