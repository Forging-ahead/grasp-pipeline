"""Goal-driven skill scheduling. This is a handwritten policy using task truth.

The original goal list is never rewritten. Each stage is checked independently,
and the environment's complete predicate is checked again after all stages.
"""
from __future__ import annotations


def normalize_goal(goal):
    return [str(x).lower() for x in goal]


def owner_of(name, object_names, regions):
    if name in object_names:
        return name
    if name in regions and regions[name].get('target') in object_names:
        return regions[name]['target']
    matches = [x for x in object_names if name.startswith(x + '_')]
    if matches:
        return max(matches, key=len)
    raise ValueError('Cannot resolve goal owner: ' + name)


def schedule(goals, object_names, regions, language=''):
    """Compose open/close, switch and placement skills without task-name cases."""
    goals = [normalize_goal(g) for g in goals]
    places, before, after = [], [], []
    for goal in goals:
        op = goal[0]
        if op in ('on', 'in') and len(goal) == 3:
            places.append(dict(skill='push' if 'push' in language.lower() else 'pick_place',
                predicate=goal, target=goal[1], destination=owner_of(goal[2], object_names, regions),
                region=goal[2], relation=op))
        elif op in ('open', 'close', 'turnon', 'turnoff') and len(goal) == 2:
            stage = dict(skill='articulate', predicate=goal,
                target=owner_of(goal[1], object_names, regions), region=goal[1], relation=op)
            # Keep an aperture open until everything destined for it is placed.
            delayed = op == 'close' and any(g[0] == 'in' and (
                g[2] == goal[1] or (goal[1] == stage['target'] and
                    owner_of(g[2],object_names,regions) == stage['target'])) for g in goals)
            (after if delayed or op == 'turnoff' else before).append(stage)
        else:
            raise ValueError('No skill for official predicate: ' + repr(goal))
    # Place a supporting object first, then put the upper object onto it.
    for stage in before:
        if stage['relation']=='open' and any(
                p['relation']=='in' and p['destination']==stage['target'] for p in places):
            stage['requires_access']=True
    ordered = []
    while places:
        ready = [x for x in places if x['destination'] not in {y['target'] for y in places}]
        if not ready:
            raise ValueError('Cyclic placement dependencies')
        for x in ready:
            ordered.append(x)
            places.remove(x)
    return before + ordered + after
