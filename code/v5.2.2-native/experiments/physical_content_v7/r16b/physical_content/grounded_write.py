"""Keep physical changes/state interaction, remove the free common offset.

For common message m this becomes (state-mean(state))*m: common physical
information still changes current video features. For varying messages the
mean update is state/message covariance. No clipping, gate or new parameter.
"""

def grounded_message(message,state):
    m=message.float();s=state.float()
    mean_m=m.mean(dim=1,keepdim=True);mean_s=s.mean(dim=1,keepdim=True)
    return (m-mean_m)+(s*m-mean_s*mean_m)
