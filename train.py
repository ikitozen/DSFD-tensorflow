from net_work import trainner
import setproctitle
setproctitle.setproctitle("ssd-like")

trainner=trainner()

trainner.train()
