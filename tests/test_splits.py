from myobridge.splits import build_splits


def membership():
    trials = []
    for participant in ('p1', 'p2', 'p3', 'p4'):
        for side in ('L', 'R'):
            for session in (1, 2):
                for action in (1, 2):
                    for repetition in (1, 2, 3, 4, 5):
                        trials.append(dict(
                            trial_id=f'{participant}-{side}-{session}-{action}-{repetition}',
                            package=f'{participant}-{side}', participant=participant,
                            side=side, session=session, action=action, repetition=repetition,
                            frames=10))
    return dict(trials=trials)


def test_protocols_are_disjoint_and_deterministic():
    first = build_splits(membership(), split_seed=0)
    second = build_splits(membership(), split_seed=0)
    assert first['protocols'] == second['protocols']
    for name, split in first['protocols'].items():
        train, test = set(split['train']), set(split['test'])
        assert train and test and not (train & test)
    cross_trial = first['protocols']['cross_trial']
    assert len(cross_trial['train']) == 4 * 2 * 2 * 2 * 3
    cross_session = first['protocols']['cross_session']
    assert all(t.split('-')[2] == '1' for t in cross_session['train'])
    assert all(t.split('-')[2] == '2' for t in cross_session['test'])
    cross_subject = first['protocols']['cross_subject']
    train_people = {t.split('-')[0] for t in cross_subject['train']}
    test_people = {t.split('-')[0] for t in cross_subject['test']}
    assert not train_people & test_people
    assert len(train_people) == 3 and len(test_people) == 1
