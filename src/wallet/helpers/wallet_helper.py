from datetime import datetime


def transaction_report(account_messages):
    transactions_resumed = []
    transactions_complete = []

    for ac in account_messages:
        if 'transaction' not in ac['payload']:
            continue  # ignore this entry, it is not a transaction

        confirmed = False if 'confirmed' not in ac else ac['confirmed']
        transaction_list = []
        for t in ac['payload']['transaction']:
            transaction_list.append(
                {
                    'incoming': t['essence']['regular']['incoming'],
                    'value': t['essence']['regular']['value'],
                }
            )

        transactions_resumed.append(
            {
                'id': ac['id'],
                'timestamp': datetime.fromtimestamp(ac['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                'confirmed': confirmed,
                'transaction_list': transaction_list,
            }
        )
        transactions_complete.append(ac)

    data = {
        "transactions_resumed": transactions_resumed,
        "transactions_complete_log": transactions_complete
    }

    return data
