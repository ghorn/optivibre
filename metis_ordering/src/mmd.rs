#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use super::{MetisGraph, OrderingError, OrderingStats, OrderingSummary, Permutation};

type Idx = isize;

pub(super) fn mmd_order(graph: &MetisGraph) -> Result<OrderingSummary, OrderingError> {
    let vertex_count = graph.vertex_count();
    let mut xadj = vec![0; vertex_count + 2];
    for vertex in 0..=vertex_count {
        xadj[vertex + 1] = checked_idx(graph.offsets[vertex] + 1, "MMD xadj")?;
    }
    let mut adjncy = vec![0; graph.neighbors.len() + 1];
    for (entry, &neighbor) in graph.neighbors.iter().enumerate() {
        adjncy[entry + 1] = checked_idx(neighbor + 1, "MMD adjncy")?;
    }

    let mut invp = vec![0; vertex_count + 5];
    let mut perm = vec![0; vertex_count + 5];
    let mut head = vec![0; vertex_count + 5];
    let mut qsize = vec![0; vertex_count + 5];
    let mut list = vec![0; vertex_count + 5];
    let mut marker = vec![0; vertex_count + 5];
    let mut nofsub = 0;

    genmmd(
        vertex_count,
        &mut xadj,
        &mut adjncy,
        &mut invp,
        &mut perm,
        1,
        &mut head,
        &mut qsize,
        &mut list,
        &mut marker,
        Idx::MAX,
        &mut nofsub,
    );

    let mut order = vec![0; vertex_count];
    for vertex in 0..vertex_count {
        let position = usize::try_from(invp[vertex + 1] - 1).map_err(|_| {
            OrderingError::Algorithm(format!(
                "METIS MMD produced invalid inverse order entry {} for vertex {vertex}",
                invp[vertex + 1]
            ))
        })?;
        if position >= vertex_count {
            return Err(OrderingError::Algorithm(format!(
                "METIS MMD inverse order entry {position} out of bounds for {vertex_count} vertices"
            )));
        }
        order[position] = vertex;
    }

    Ok(OrderingSummary {
        permutation: Permutation::new(order)?,
        stats: OrderingStats {
            connected_components: 0,
            separator_calls: 0,
            leaf_calls: usize::from(vertex_count > 0),
            separator_vertices: 0,
            max_separator_size: 0,
        },
    })
}

fn checked_idx(value: usize, what: &str) -> Result<Idx, OrderingError> {
    Idx::try_from(value)
        .map_err(|_| OrderingError::Algorithm(format!("{what} value {value} exceeds idx_t range")))
}

fn as_usize(value: Idx) -> usize {
    usize::try_from(value).expect("METIS MMD internal index must be non-negative")
}

fn genmmd(
    neqns: usize,
    xadj: &mut [Idx],
    adjncy: &mut [Idx],
    invp: &mut [Idx],
    perm: &mut [Idx],
    delta: Idx,
    head: &mut [Idx],
    qsize: &mut [Idx],
    list: &mut [Idx],
    marker: &mut [Idx],
    maxint: Idx,
    ncsub: &mut Idx,
) {
    if neqns == 0 {
        return;
    }

    *ncsub = 0;
    mmdint(neqns, xadj, adjncy, head, invp, perm, qsize, list, marker);

    let neqns_idx = checked_idx(neqns, "MMD neqns").expect("validated graph size");
    let mut num = 1;

    let mut nextmd = head[1];
    while nextmd > 0 {
        let mdeg_node = nextmd;
        nextmd = invp[as_usize(mdeg_node)];
        marker[as_usize(mdeg_node)] = maxint;
        invp[as_usize(mdeg_node)] = -num;
        num += 1;
    }

    if num <= neqns_idx {
        let mut tag = 1;
        head[1] = 0;
        let mut mdeg = 2;

        loop {
            while head[as_usize(mdeg)] <= 0 {
                mdeg += 1;
            }
            let mdlmt = neqns_idx.min(mdeg + delta);
            let mut ehead = 0;

            loop {
                let mut mdeg_node = head[as_usize(mdeg)];
                while mdeg_node <= 0 {
                    mdeg += 1;
                    if mdeg > mdlmt {
                        break;
                    }
                    mdeg_node = head[as_usize(mdeg)];
                }
                if mdeg > mdlmt {
                    break;
                }

                let next = invp[as_usize(mdeg_node)];
                head[as_usize(mdeg)] = next;
                if next > 0 {
                    perm[as_usize(next)] = -mdeg;
                }
                invp[as_usize(mdeg_node)] = -num;
                *ncsub += mdeg + qsize[as_usize(mdeg_node)] - 2;
                if num + qsize[as_usize(mdeg_node)] > neqns_idx {
                    mmdnum(neqns, perm, invp, qsize);
                    return;
                }

                tag += 1;
                if tag >= maxint {
                    tag = 1;
                    for entry in marker.iter_mut().take(neqns + 1).skip(1) {
                        if *entry < maxint {
                            *entry = 0;
                        }
                    }
                }

                mmdelm(
                    mdeg_node, xadj, adjncy, head, invp, perm, qsize, list, marker, maxint, tag,
                );

                num += qsize[as_usize(mdeg_node)];
                list[as_usize(mdeg_node)] = ehead;
                ehead = mdeg_node;
                if delta < 0 {
                    break;
                }
            }

            if num > neqns_idx {
                break;
            }
            mmdupd(
                ehead, neqns, xadj, adjncy, delta, &mut mdeg, head, invp, perm, qsize, list,
                marker, maxint, &mut tag,
            );
        }
    }

    mmdnum(neqns, perm, invp, qsize);
}

fn mmdelm(
    mdeg_node: Idx,
    xadj: &mut [Idx],
    adjncy: &mut [Idx],
    head: &mut [Idx],
    forward: &mut [Idx],
    backward: &mut [Idx],
    qsize: &mut [Idx],
    list: &mut [Idx],
    marker: &mut [Idx],
    maxint: Idx,
    tag: Idx,
) {
    marker[as_usize(mdeg_node)] = tag;
    let istart = xadj[as_usize(mdeg_node)];
    let istop = xadj[as_usize(mdeg_node + 1)] - 1;

    let mut element = 0;
    let mut rloc = istart;
    let mut rlmt = istop;
    if istart <= istop {
        for i in as_usize(istart)..=as_usize(istop) {
            let nabor = adjncy[i];
            if nabor == 0 {
                break;
            }
            if marker[as_usize(nabor)] < tag {
                marker[as_usize(nabor)] = tag;
                if forward[as_usize(nabor)] < 0 {
                    list[as_usize(nabor)] = element;
                    element = nabor;
                } else {
                    adjncy[as_usize(rloc)] = nabor;
                    rloc += 1;
                }
            }
        }
    }

    while element > 0 {
        adjncy[as_usize(rlmt)] = -element;
        let mut link = element;

        'merge_element: loop {
            let jstart = xadj[as_usize(link)];
            let jstop = xadj[as_usize(link + 1)] - 1;
            if jstart <= jstop {
                for j in as_usize(jstart)..=as_usize(jstop) {
                    let node = adjncy[j];
                    link = -node;
                    if node < 0 {
                        continue 'merge_element;
                    }
                    if node == 0 {
                        break 'merge_element;
                    }
                    if marker[as_usize(node)] < tag && forward[as_usize(node)] >= 0 {
                        marker[as_usize(node)] = tag;
                        while rloc >= rlmt {
                            link = -adjncy[as_usize(rlmt)];
                            rloc = xadj[as_usize(link)];
                            rlmt = xadj[as_usize(link + 1)] - 1;
                        }
                        adjncy[as_usize(rloc)] = node;
                        rloc += 1;
                    }
                }
            }
            break;
        }
        element = list[as_usize(element)];
    }
    if rloc <= rlmt {
        adjncy[as_usize(rloc)] = 0;
    }

    let mut link = mdeg_node;
    'reachable: loop {
        let istart = xadj[as_usize(link)];
        let istop = xadj[as_usize(link + 1)] - 1;
        if istart <= istop {
            for i in as_usize(istart)..=as_usize(istop) {
                let rnode = adjncy[i];
                link = -rnode;
                if rnode < 0 {
                    continue 'reachable;
                }
                if rnode == 0 {
                    return;
                }

                let pvnode = backward[as_usize(rnode)];
                if pvnode != 0 && pvnode != -maxint {
                    let nxnode = forward[as_usize(rnode)];
                    if nxnode > 0 {
                        backward[as_usize(nxnode)] = pvnode;
                    }
                    if pvnode > 0 {
                        forward[as_usize(pvnode)] = nxnode;
                    }
                    let npv = -pvnode;
                    if pvnode < 0 {
                        head[as_usize(npv)] = nxnode;
                    }
                }

                let jstart = xadj[as_usize(rnode)];
                let jstop = xadj[as_usize(rnode + 1)] - 1;
                let mut xqnbr = jstart;
                if jstart <= jstop {
                    for j in as_usize(jstart)..=as_usize(jstop) {
                        let nabor = adjncy[j];
                        if nabor == 0 {
                            break;
                        }
                        if marker[as_usize(nabor)] < tag {
                            adjncy[as_usize(xqnbr)] = nabor;
                            xqnbr += 1;
                        }
                    }
                }

                let nqnbrs = xqnbr - jstart;
                if nqnbrs <= 0 {
                    qsize[as_usize(mdeg_node)] += qsize[as_usize(rnode)];
                    qsize[as_usize(rnode)] = 0;
                    marker[as_usize(rnode)] = maxint;
                    forward[as_usize(rnode)] = -mdeg_node;
                    backward[as_usize(rnode)] = -maxint;
                } else {
                    forward[as_usize(rnode)] = nqnbrs + 1;
                    backward[as_usize(rnode)] = 0;
                    adjncy[as_usize(xqnbr)] = mdeg_node;
                    xqnbr += 1;
                    if xqnbr <= jstop {
                        adjncy[as_usize(xqnbr)] = 0;
                    }
                }
            }
        }
        return;
    }
}

fn mmdint(
    neqns: usize,
    xadj: &mut [Idx],
    _adjncy: &mut [Idx],
    head: &mut [Idx],
    forward: &mut [Idx],
    backward: &mut [Idx],
    qsize: &mut [Idx],
    list: &mut [Idx],
    marker: &mut [Idx],
) {
    for node in 1..=neqns {
        head[node] = 0;
        qsize[node] = 1;
        marker[node] = 0;
        list[node] = 0;
    }

    for node in 1..=neqns {
        let ndeg = xadj[node + 1] - xadj[node] + 1;
        let fnode = head[as_usize(ndeg)];
        forward[node] = fnode;
        head[as_usize(ndeg)] = checked_idx(node, "MMD node").expect("validated graph size");
        if fnode > 0 {
            backward[as_usize(fnode)] =
                checked_idx(node, "MMD node").expect("validated graph size");
        }
        backward[node] = -ndeg;
    }
}

fn mmdnum(neqns: usize, perm: &mut [Idx], invp: &mut [Idx], qsize: &mut [Idx]) {
    for node in 1..=neqns {
        let nqsize = qsize[node];
        if nqsize <= 0 {
            perm[node] = invp[node];
        }
        if nqsize > 0 {
            perm[node] = -invp[node];
        }
    }

    for node in 1..=neqns {
        if perm[node] <= 0 {
            let mut father = checked_idx(node, "MMD node").expect("validated graph size");
            while perm[as_usize(father)] <= 0 {
                father = -perm[as_usize(father)];
            }

            let root = father;
            let num = perm[as_usize(root)] + 1;
            invp[node] = -num;
            perm[as_usize(root)] = num;

            father = checked_idx(node, "MMD node").expect("validated graph size");
            let mut nextf = -perm[as_usize(father)];
            while nextf > 0 {
                perm[as_usize(father)] = -root;
                father = nextf;
                nextf = -perm[as_usize(father)];
            }
        }
    }

    for node in 1..=neqns {
        let num = -invp[node];
        invp[node] = num;
        perm[as_usize(num)] = checked_idx(node, "MMD node").expect("validated graph size");
    }
}

fn mmdupd(
    ehead: Idx,
    neqns: usize,
    xadj: &mut [Idx],
    adjncy: &mut [Idx],
    delta: Idx,
    mdeg: &mut Idx,
    head: &mut [Idx],
    forward: &mut [Idx],
    backward: &mut [Idx],
    qsize: &mut [Idx],
    list: &mut [Idx],
    marker: &mut [Idx],
    maxint: Idx,
    tag: &mut Idx,
) {
    let mdeg0 = *mdeg + delta;
    let mut element = ehead;

    while element > 0 {
        let mut mtag = *tag + mdeg0;
        if mtag >= maxint {
            *tag = 1;
            for entry in marker.iter_mut().take(neqns + 1).skip(1) {
                if *entry < maxint {
                    *entry = 0;
                }
            }
            mtag = *tag + mdeg0;
        }

        let mut q2head = 0;
        let mut qxhead = 0;
        let mut deg0 = 0;
        let mut link = element;

        'element_scan: loop {
            let istart = xadj[as_usize(link)];
            let istop = xadj[as_usize(link + 1)] - 1;
            if istart <= istop {
                for i in as_usize(istart)..=as_usize(istop) {
                    let enode = adjncy[i];
                    link = -enode;
                    if enode < 0 {
                        continue 'element_scan;
                    }
                    if enode == 0 {
                        break 'element_scan;
                    }
                    if qsize[as_usize(enode)] != 0 {
                        deg0 += qsize[as_usize(enode)];
                        marker[as_usize(enode)] = mtag;
                        if backward[as_usize(enode)] == 0 {
                            if forward[as_usize(enode)] != 2 {
                                list[as_usize(enode)] = qxhead;
                                qxhead = enode;
                            } else {
                                list[as_usize(enode)] = q2head;
                                q2head = enode;
                            }
                        }
                    }
                }
            }
            break;
        }

        let mut enode = q2head;
        let mut iq2 = true;

        loop {
            while enode > 0 {
                if backward[as_usize(enode)] == 0 {
                    *tag += 1;
                    let mut deg = deg0;

                    if iq2 {
                        let istart = xadj[as_usize(enode)];
                        let mut nabor = adjncy[as_usize(istart)];
                        if nabor == element {
                            nabor = adjncy[as_usize(istart + 1)];
                        }
                        let mut link = nabor;
                        if forward[as_usize(nabor)] >= 0 {
                            deg += qsize[as_usize(nabor)];
                        } else {
                            'second_element: loop {
                                let jstart = xadj[as_usize(link)];
                                let jstop = xadj[as_usize(link + 1)] - 1;
                                if jstart <= jstop {
                                    for j in as_usize(jstart)..=as_usize(jstop) {
                                        let node = adjncy[j];
                                        link = -node;
                                        if node == enode {
                                            continue;
                                        }
                                        if node < 0 {
                                            continue 'second_element;
                                        }
                                        if node == 0 {
                                            break 'second_element;
                                        }
                                        if qsize[as_usize(node)] != 0 {
                                            if marker[as_usize(node)] < *tag {
                                                marker[as_usize(node)] = *tag;
                                                deg += qsize[as_usize(node)];
                                            } else if backward[as_usize(node)] == 0 {
                                                if forward[as_usize(node)] == 2 {
                                                    qsize[as_usize(enode)] += qsize[as_usize(node)];
                                                    qsize[as_usize(node)] = 0;
                                                    marker[as_usize(node)] = maxint;
                                                    forward[as_usize(node)] = -enode;
                                                    backward[as_usize(node)] = -maxint;
                                                } else if backward[as_usize(node)] == 0 {
                                                    backward[as_usize(node)] = -maxint;
                                                }
                                            }
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    } else {
                        let istart = xadj[as_usize(enode)];
                        let istop = xadj[as_usize(enode + 1)] - 1;
                        if istart <= istop {
                            for i in as_usize(istart)..=as_usize(istop) {
                                let nabor = adjncy[i];
                                if nabor == 0 {
                                    break;
                                }
                                if marker[as_usize(nabor)] < *tag {
                                    marker[as_usize(nabor)] = *tag;
                                    let mut link = nabor;
                                    if forward[as_usize(nabor)] >= 0 {
                                        deg += qsize[as_usize(nabor)];
                                    } else {
                                        'eliminated_nabor: loop {
                                            let jstart = xadj[as_usize(link)];
                                            let jstop = xadj[as_usize(link + 1)] - 1;
                                            if jstart <= jstop {
                                                for j in as_usize(jstart)..=as_usize(jstop) {
                                                    let node = adjncy[j];
                                                    link = -node;
                                                    if node < 0 {
                                                        continue 'eliminated_nabor;
                                                    }
                                                    if node == 0 {
                                                        break 'eliminated_nabor;
                                                    }
                                                    if marker[as_usize(node)] < *tag {
                                                        marker[as_usize(node)] = *tag;
                                                        deg += qsize[as_usize(node)];
                                                    }
                                                }
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    deg = deg - qsize[as_usize(enode)] + 1;
                    let fnode = head[as_usize(deg)];
                    forward[as_usize(enode)] = fnode;
                    backward[as_usize(enode)] = -deg;
                    if fnode > 0 {
                        backward[as_usize(fnode)] = enode;
                    }
                    head[as_usize(deg)] = enode;
                    if deg < *mdeg {
                        *mdeg = deg;
                    }
                }
                enode = list[as_usize(enode)];
            }

            if iq2 {
                enode = qxhead;
                iq2 = false;
            } else {
                break;
            }
        }

        *tag = mtag;
        element = list[as_usize(element)];
    }
}
