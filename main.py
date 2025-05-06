import yaml
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

from plot import (
    plot_vote_heatmap,
    draw_vote_flow_cum_rounds_weighted,
    draw_vote_flow_graph,
    plot_all_clusterings,
    plot_vote_share,
    plot_breakfast_groups,
)

from vtypes import (
    GameData,
    MurderEvent,
    ShieldEvent,
    DaggerEvent,
    RoundTableEvent,
    BreakfastEvent,
    BreakfastOrdering,
    GameRound,
    GameEvent,
)
from typing import List

from cluster import (
    girvan_newman,
    infomap_communities,
    label_propagation,
    spectral,
)

def check_who_was_banned(votes: dict[str,str], weights: dict[str, float], default_weight: float = 1.0):
    totals = defaultdict(float)
    for voter, target in votes.items():
        w = weights.get(voter, default_weight)
        totals[target] += w

    if not totals:
        raise ValueError("No votes to tally")

    # 2) Find the maximum total
    max_votes = max(totals.values())

    # 3) Collect all candidates with that max
    winners = [cand for cand, score in totals.items() if score == max_votes]

    # Return single name or tie list
    return [winners[0]] if len(winners) == 1 else winners

def check(game_data: GameData):
    part_left = set(game_data.participants)
    for r in game_data.rounds:
        voting_power = {}
        next_voters,voters = None,None
        for event in r.events:
            if event.type == 'murder':
                try:
                    part_left.remove(event.name)
                except Exception as e:
                    print(f"trying to remove {event.name} from {part_left}")
                    raise e
            elif event.type == 'round table':
                if not voters: voters = set(part_left)
                else: next_voters = set(part_left)
                assert event.votes.keys() == voters
                banned = check_who_was_banned(event.votes, {})
                if len(banned) == 1:
                    # voting worked out
                    part_left.remove(banned[0])
                elif not next_voters:
                    # first tie -> tied people are not allowed to vote
                    for b in banned: voters.remove(b)
                else:
                    # second tie -> all are allowed to vote again
                    voters = next_voters
            elif event.type == 'breakfast':
                assert sum(map(lambda x: len(x.group), event.ordering)) == len(part_left)
                assert set().union(*map(lambda x: set(x.group), event.ordering)) == part_left
            elif event.type == 'shield':
                for n in event.names:
                    assert n in part_left
            elif event.type == 'dagger':
                for n in event.names:
                    assert n in part_left
                    voting_power[n] = 2
            else:
                assert False, f'Invalid type ({event['type']}) in round found'


# Function to load the YAML file
def load_game_data(file_path) -> GameData:
    with open(file_path, 'r') as file:
        raw = yaml.load(file, Loader=yaml.FullLoader)

    rounds = []
    for round_raw in raw['rounds']:
        events = []
        for event in round_raw['events']:
            if event['type'] == 'murder':
                if 'name' in event:
                    events.append(MurderEvent(**event))
            elif event['type'] == 'shield':
                events.append(ShieldEvent(**event))
            elif event['type'] == 'dagger':
                events.append(DaggerEvent(**event))
            elif event['type'] == 'round table':
                events.append(RoundTableEvent(**event))
            elif event['type'] == 'breakfast':
                events.append(BreakfastEvent(type=event['type'], ordering=[BreakfastOrdering(**i) for i in event['ordering']]))
        rounds.append(GameRound(events=events))

    return GameData(
        participants=raw['participants'],
        traitors=raw['traitors'],
        rounds=rounds
    )


# Build vote matrix
def build_vote_matrix(game_data: GameData, until_round: int|None):
    participants = game_data.participants
    name_to_index = {name: i for i, name in enumerate(participants)}
    vote_matrix = np.zeros((len(participants), len(participants)), dtype=int)

    for rnd in game_data.rounds[:until_round]:
        for event in rnd.events:
            if event.type == 'round table':
                for voter, target in event.votes.items():
                    i = name_to_index[voter]
                    j = name_to_index[target]
                    vote_matrix[i][j] += 1
                break # only consider the first round table vote here

    return vote_matrix, participants


def vote_flow_graph(game_data: GameData, round_index:int):
    round_data:List[GameEvent] = game_data.rounds[round_index].events
    participants = set()
    for event in round_data:
        if event.type == 'round table':
            participants.union(event.votes.keys())
            break # only consider the first round table vote here

    G = nx.DiGraph()
    G.add_nodes_from(participants)

    immune = set()

    # Extract events
    for event in round_data:
        if event.type == 'shield':
            immune.update(event.names)
        elif event.type == 'round table':
            for voter, target in event.votes.items():
                G.add_edge(voter, target)
            break # only consider the first round table vote here
    return G, immune


def vote_flow_cum_rounds_weighted(game_data:GameData, until_round:int|None):
    rounds = game_data.rounds[:until_round]
    participants = set()
    for rnd in rounds:
        for event in rnd.events:
            if event.type == 'round table':
                participants.union(event.votes.keys())
            break # only consider the first round table vote here

    G = nx.DiGraph()
    G.add_nodes_from(participants)

    vote_counts = defaultdict(int)

    # Count repeated votes across rounds
    for rnd in rounds:
        for event in rnd.events:
            if event.type == 'round table':
                for voter, target in event.votes.items():
                    vote_counts[(voter, target)] += 1
                    G.add_edge(voter, target)
                break # only consider the first round table vote in each round
    return G, vote_counts


def build_cum_vote_share_data(game_data, until_round:int|None):
    rounds = game_data.rounds[:until_round]
    participants = game_data.participants

    # Initialize DataFrame with zeros
    vote_share = pd.DataFrame(0, index=range(1, len(rounds)+1), columns=participants)

    for round_idx, rnd in enumerate(rounds, start=1):
        # Count votes for this round
        for event in rnd.events:
            if event.type == 'round table':
                for target in event.votes.values():
                    vote_share.loc[round_idx, target] += 1
                break # only consider the first round table vote in each round

    return vote_share

def build_vote_share_data(game_data, round_index: int):
    participants = game_data.participants
    rounds = game_data.rounds

    # Initialize DataFrame with zeros for the specified round
    vote_share = pd.DataFrame(0, index=[round_index], columns=participants)

    # Retrieve the specified round (adjusting for 0-based indexing)
    rnd = rounds[round_index - 1]

    # Count votes for the specified round
    for event in rnd.events:
        if event.type == 'round table':
            for target in event.votes.values():
                vote_share.loc[round_index, target] += 1
            break # only consider the first round table vote in each round

    return vote_share

def build_breakfast_grouping_table(
    game_data: GameData,
    until_round: int|None = None
) -> pd.DataFrame:
    """
    Scans each round's BreakfastEvent up to `until_round`, counts how many traitors
    are in each breakfast group, and returns a DataFrame indexed by group (1, 2, ...)
    with columns for each round.

    Parameters
    ----------
    game_data : GameData
        Your full game data.
    until_round : int or None, default None
        If int, only include rounds with index <= until_round.
        If None, include all rounds.

    Returns
    -------
    pd.DataFrame
        Pivot table where rows are group indices, columns are round numbers,
        and values are traitor counts (NaN if that group didn't exist).
    """
    records = []
    for rnd_idx, game_round in enumerate(game_data.rounds):
        # stop if we've passed the cutoff
        if until_round is not None and rnd_idx > until_round:
            break

        # find the breakfast event
        breakfast = next(
            (e for e in game_round.events if isinstance(e, BreakfastEvent)),
            None
        )
        if breakfast is None:
            # no breakfast this round; skip
            continue

        # for each group in the breakfast ordering
        for grp_idx, grp_dict in enumerate(breakfast.ordering):
            member_names = grp_dict.group
            # count traitors in this group
            traitor_count = sum(name in game_data.traitors for name in member_names)
            records.append({
                'round': rnd_idx,
                'group': grp_idx + 1,           # 1-based group index
                'traitor_count': traitor_count
            })

    # build DataFrame and pivot, using NaN for missing
    df = pd.DataFrame.from_records(records)
    table = df.pivot_table(
        index='group',
        columns='round',
        values='traitor_count',
        aggfunc='sum',
        fill_value=np.nan
    )

    # sort rows by group index
    table = table.sort_index(axis=0)
    return table

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser('check')

    heatmap = subparsers.add_parser('heatmap')
    heatmap.add_argument('-f', '--final', action='store_true')
    heatmap.add_argument('-r', '--round', type=int)

    flow = subparsers.add_parser('flow')
    flow.add_argument('-c', '--cumulated', action='store_true')
    flow.add_argument('-f', '--final', action='store_true')
    flow.add_argument('-r', '--round', type=int)

    cluster = subparsers.add_parser('cluster')
    cluster.add_argument('-c', '--cumulated', action='store_true')
    cluster.add_argument('-f', '--final', action='store_true')
    cluster.add_argument('-r', '--round', type=int)

    votes = subparsers.add_parser('votes')
    votes.add_argument('-c', '--cumulated', action='store_true')
    votes.add_argument('-f', '--final', action='store_true')
    votes.add_argument('-r', '--round', type=int)

    votes = subparsers.add_parser('breakfast')
    votes.add_argument('-f', '--final', action='store_true')
    votes.add_argument('-r', '--round', type=int)

    parser.add_argument('fn')
    parser.add_argument('-o', '--output')

    return parser.parse_args()

# Main
if __name__ == '__main__':
    args = parse_args()

    game_data = load_game_data(args.fn)

    if args.command == 'check':
        check(game_data)

    elif args.command == 'heatmap':
        if args.round is not None:
            ds = (build_vote_matrix(game_data, until_round=args.round),)
        elif args.final:
            ds = (build_vote_matrix(game_data, until_round=None),)
        else:
            ds = map(lambda i: build_vote_matrix(game_data, until_round=i+1), range(len(game_data.rounds)))

        for i, d in enumerate(ds):
            plt = plot_vote_heatmap(d[0], d[1])
            if not args.output:
                plt.show()
            else:
                plt.savefig(
                    args.output.format(i),
                    dpi=300,
                    bbox_inches='tight',
                )

    elif args.command == 'flow':
        if args.cumulated:
            if args.round is not None:
                ds = (vote_flow_cum_rounds_weighted(game_data, until_round=args.round),)
            elif args.final:
                ds = (vote_flow_cum_rounds_weighted(game_data, until_round=None),)
            else:
                ds = map(lambda i: vote_flow_cum_rounds_weighted(game_data, until_round=i+1), range(len(game_data.rounds)))
            for i,d in enumerate(ds):
                plt = draw_vote_flow_cum_rounds_weighted(game_data, d[0], d[1])
                if not args.output:
                    plt.show()
                else:
                    plt.savefig(
                        args.output.format(i),
                        dpi=300,
                        bbox_inches='tight',
                    )

        else:
            if args.round is not None:
                ds = (vote_flow_graph(game_data, round_index=args.round),)
            elif args.final:
                ds = (vote_flow_graph(game_data, round_index=len(game_data.rounds)-1),)
            else:
                ds = map(lambda i: vote_flow_graph(game_data, round_index=i), range(len(game_data.rounds)))
            for i, d in enumerate(ds):
                plt = draw_vote_flow_graph(game_data, d[0], d[1])
                if not args.output:
                    plt.show()
                else:
                    plt.savefig(
                        args.output.format(i),
                        dpi=300,
                        bbox_inches='tight',
                    )

    elif args.command == 'cluster':
        if args.cumulated:
            if args.round is not None:
                ds = (vote_flow_cum_rounds_weighted(game_data, until_round=args.round),)
            elif args.final:
                ds = (vote_flow_cum_rounds_weighted(game_data, until_round=None),)
            else:
                ds = map(lambda i: vote_flow_cum_rounds_weighted(game_data, until_round=i+1), range(len(game_data.rounds)))
        else:
            if args.round is not None:
                ds = (vote_flow_graph(game_data, round_index=args.round),)
            elif args.final:
                ds = (vote_flow_graph(game_data, round_index=len(game_data.rounds)-1),)
            else:
                ds = map(lambda i: vote_flow_graph(game_data, round_index=i), range(len(game_data.rounds)))

        for i, d in enumerate(ds):
            plt = plot_all_clusterings(
                game_data,
                d[0],
                methods = [
                    ("Girvan–Newman", girvan_newman),
                    ("Infomap", infomap_communities),
                    ("Label Propagation", label_propagation),
                    ("Spectral Clustering", spectral),
                ],
            )
            if not args.output:
                plt.show()
            else:
                plt.savefig(
                    args.output.format(i),
                    dpi=300,
                    bbox_inches='tight',
                )

    elif args.command == 'votes':
        if args.cumulated:
            ds = (build_cum_vote_share_data(game_data, until_round=None),)
        else:
            if args.round is not None:
                ds = (build_vote_share_data(game_data, round_index=args.round),)
            elif args.final:
                ds = (build_vote_share_data(game_data, round_index=len(game_data.rounds)),)
            else:
                ds = map(lambda i: build_vote_share_data(game_data, round_index=i+1), range(len(game_data.rounds)))
        for i, d in enumerate(ds):
            plt = plot_vote_share(d)
            if not args.output:
                plt.show()
            else:
                plt.savefig(
                    args.output.format(i),
                    dpi=300,
                    bbox_inches='tight',
                )

    elif args.command == 'breakfast':
        if args.round is not None:
            ds = (build_breakfast_grouping_table(game_data, until_round=args.round),)
        elif args.final:
            ds = (build_breakfast_grouping_table(game_data, until_round=len(game_data.rounds)),)
        else:
            ds = map(lambda i: build_breakfast_grouping_table(game_data, until_round=i), range(len(game_data.rounds)))

        for i, d in enumerate(ds):
            plt = plot_breakfast_groups(d)
            if not args.output:
                plt.show()
            else:
                plt.savefig(
                    args.output.format(i),
                    dpi=300,
                    bbox_inches='tight',
                )
