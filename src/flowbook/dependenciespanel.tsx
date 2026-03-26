/**
 * Dependencies Panel - ReactFlow graph showing cell-to-cell dependencies.
 *
 * Nodes = cells (labeled @A, @B, ...) in vertical program order.
 * Edges:
 *   - Program order: straight gray arrows centered vertically
 *   - Write->Read: curved blue arrows (shown on hover only)
 *   - Write->Write / Read->Write: curved red dashed arrows (always shown)
 *
 * Hover behavior:
 *   - Data dependency edges are hidden by default
 *   - When hovering a node, its dependency edges appear
 *   - Error edges (write_write, read_write) are always visible
 */

import { Widget } from '@lumino/widgets';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import {
  ReactFlow,
  ReactFlowProvider,
  Controls,
  BaseEdge,
  MarkerType,
  type Node,
  type Edge,
  type EdgeProps,
  type NodeMouseHandler,
  Position,
  Handle
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import {
  IReadLoc,
  IWriteLoc,
  IPredicateViolation,
  writeConflictsRead,
  formatReadLoc
} from './types';

// ============================================================================
// Data model
// ============================================================================

export interface ICellGraphData {
  cellId: string;
  index: number;
  label: string;
  readLocs: IReadLoc[];
  writeLocs: IWriteLoc[];
  isStale: boolean;
  isExecuted: boolean;
  hasError: boolean;
  /** Predicate violations on this cell (from flowbook_violations metadata) */
  violations: IPredicateViolation[];
}

type DependencyEdgeKind = 'program_order' | 'data_dependency' | 'violation';

interface IDependencyEdge {
  sourceId: string;
  targetId: string;
  kind: DependencyEdgeKind;
  locations: string[];
  span: number;
  lane: number;
}

// ============================================================================
// Edge computation
// ============================================================================

function computeEdges(cells: ICellGraphData[]): IDependencyEdge[] {
  const depEdges: IDependencyEdge[] = [];

  for (let i = 0; i < cells.length; i++) {
    if (!cells[i].isExecuted) {
      continue;
    }
    for (let j = i + 1; j < cells.length; j++) {
      if (!cells[j].isExecuted) {
        continue;
      }

      const span = j - i;

      // Write->Read
      const wrLocs: string[] = [];
      const wrSeen = new Set<string>();
      for (const w of cells[i].writeLocs) {
        for (const r of cells[j].readLocs) {
          if (writeConflictsRead(w, r)) {
            const label = formatReadLoc(r);
            if (!wrSeen.has(label)) {
              wrSeen.add(label);
              wrLocs.push(label);
            }
          }
        }
      }
      if (wrLocs.length > 0) {
        depEdges.push({
          sourceId: cells[i].cellId,
          targetId: cells[j].cellId,
          kind: 'data_dependency',
          locations: wrLocs,
          span,
          lane: 0
        });
      }
    }
  }

  // Generate edges from predicate violations (cells whose execution was
  // rejected may have no read/write metadata, but do have violation info)
  const cellIndexMap = new Map<string, number>();
  for (let i = 0; i < cells.length; i++) {
    cellIndexMap.set(cells[i].cellId, i);
  }

  const violationEdgeKeys = new Set<string>();
  for (const cell of cells) {
    for (const v of cell.violations) {
      if (!v.causer_cell) {
        continue;
      }
      // Normalize causer_cell (may have @ prefix)
      const causerId = v.causer_cell.startsWith('@')
        ? v.causer_cell.slice(1)
        : v.causer_cell;

      const cellIdx = cellIndexMap.get(cell.cellId);
      const causerIdx = cellIndexMap.get(causerId);
      if (cellIdx === undefined || causerIdx === undefined) {
        continue;
      }

      // Edge direction depends on the predicate:
      //   no_write_after_read: violating cell wrote, causer cell read → edge from violator to causer
      //   no_read_before_write: violating cell read, causer cell wrote → edge from causer to violator
      //   no_read_and_write: self-conflict on the violating cell
      let sourceId: string;
      let targetId: string;
      if (v.predicate === 'no_write_after_read') {
        sourceId = cell.cellId; // writer
        targetId = causerId; // reader
      } else if (v.predicate === 'no_read_before_write') {
        sourceId = causerId; // writer below
        targetId = cell.cellId; // reader above
      } else {
        sourceId = cell.cellId;
        targetId = causerId;
      }

      const key = `${sourceId}-${targetId}-violation`;
      if (violationEdgeKeys.has(key)) {
        continue;
      }
      violationEdgeKeys.add(key);

      const srcIdx = cellIndexMap.get(sourceId) ?? 0;
      const tgtIdx = cellIndexMap.get(targetId) ?? 0;
      const span = Math.abs(srcIdx - tgtIdx);

      depEdges.push({
        sourceId,
        targetId,
        kind: 'violation',
        locations: v.locations,
        span: span || 1,
        lane: 0
      });
    }
  }

  // Assign lanes: sort by span so shorter edges get inner lanes
  depEdges.sort((a, b) => a.span - b.span);
  for (let i = 0; i < depEdges.length; i++) {
    depEdges[i].lane = i + 1;
  }

  // Program order edges
  const allEdges: IDependencyEdge[] = [];
  for (let i = 0; i < cells.length - 1; i++) {
    allEdges.push({
      sourceId: cells[i].cellId,
      targetId: cells[i + 1].cellId,
      kind: 'program_order',
      locations: [],
      span: 1,
      lane: 0
    });
  }

  return allEdges.concat(depEdges);
}

// ============================================================================
// Custom edge: cubic bezier that bulges right
// ============================================================================

function BulgeEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  style,
  markerEnd,
  label,
  labelStyle,
  data
}: EdgeProps) {
  const bulge = (data?.bulge as number) || 50;

  const cpX = Math.max(sourceX, targetX) + bulge;
  const cp1Y = sourceY + (targetY - sourceY) * 0.25;
  const cp2Y = sourceY + (targetY - sourceY) * 0.75;

  const path = `M ${sourceX},${sourceY} C ${cpX},${cp1Y} ${cpX},${cp2Y} ${targetX},${targetY}`;

  // Compute the actual midpoint of the cubic bezier at t=0.5
  // B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
  // At t=0.5: coefficients are 0.125, 0.375, 0.375, 0.125
  const midX = 0.125 * sourceX + 0.375 * cpX + 0.375 * cpX + 0.125 * targetX;
  const midY = 0.125 * sourceY + 0.375 * cp1Y + 0.375 * cp2Y + 0.125 * targetY;

  const labelStr = label ? String(label) : '';
  const textWidth = labelStr.length * 6.5;
  const padX = 4;
  const padY = 3;

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={style}
        markerEnd={markerEnd as string}
      />
      {labelStr && (
        <g transform={`translate(${midX}, ${midY})`}>
          <rect
            x={-textWidth / 2 - padX}
            y={-8 - padY}
            width={textWidth + padX * 2}
            height={16 + padY * 2}
            rx={3}
            ry={3}
            fill="white"
            fillOpacity={0.95}
            stroke="var(--jp-border-color2, #ddd)"
            strokeWidth={0.5}
          />
          <text
            style={labelStyle as React.CSSProperties}
            textAnchor="middle"
            dominantBaseline="central"
            className="react-flow__edge-text"
          >
            {labelStr}
          </text>
        </g>
      )}
    </>
  );
}

const edgeTypes = { bulge: BulgeEdge };

// ============================================================================
// Graph builder
// ============================================================================

const NODE_SPACING = 80;
const NODE_X = 40;
const LANE_BASE = 40;
const LANE_STEP = 25;

/**
 * Node with centered top/bottom handles for program order,
 * and right-side handles for dependency edges.
 */
function CellNode({ data }: { data: { label: string; className: string } }) {
  return (
    <div className={data.className}>
      <Handle
        type="target"
        position={Position.Top}
        id="center-top"
        style={{ opacity: 0 }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="center-bottom"
        style={{ opacity: 0 }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="right-source"
        style={{ opacity: 0 }}
      />
      <Handle
        type="target"
        position={Position.Right}
        id="right-target"
        style={{ opacity: 0 }}
      />
      {data.label}
    </div>
  );
}

const nodeTypes = { cell: CellNode };

function formatEdgeLabel(locations: string[]): string {
  if (locations.length === 0) {
    return '';
  }
  if (locations.length <= 3) {
    return locations.join(', ');
  }
  return locations.slice(0, 3).join(', ') + ', ...';
}

const ARROW_GRAY = {
  type: MarkerType.ArrowClosed,
  color: '#999',
  width: 12,
  height: 12
};

const ARROW_BLUE = {
  type: MarkerType.ArrowClosed,
  color: '#2472c8',
  width: 14,
  height: 14
};

const ARROW_RED = {
  type: MarkerType.ArrowClosed,
  color: '#dc3545',
  width: 14,
  height: 14
};

/**
 * Build all nodes and edges. Edges are tagged with data.edgeKind
 * so the FlowGraph component can filter by hover state.
 */
function buildGraph(cellData: ICellGraphData[]): {
  nodes: Node[];
  allEdges: Edge[];
} {
  const nodes: Node[] = cellData.map((cell, i) => {
    let className = 'flowbook-dep-node';
    if (cell.hasError) {
      className += ' flowbook-dep-node-error';
    } else if (cell.isStale) {
      className += ' flowbook-dep-node-stale';
    } else if (!cell.isExecuted) {
      className += ' flowbook-dep-node-unexecuted';
    }

    return {
      id: cell.cellId,
      type: 'cell',
      position: { x: NODE_X, y: i * NODE_SPACING },
      data: { label: cell.label, className }
    };
  });

  const depEdges = computeEdges(cellData);

  const allEdges: Edge[] = depEdges.map(dep => {
    const baseId = `${dep.sourceId}-${dep.targetId}-${dep.kind}`;

    if (dep.kind === 'program_order') {
      return {
        id: baseId,
        source: dep.sourceId,
        target: dep.targetId,
        sourceHandle: 'center-bottom',
        targetHandle: 'center-top',
        type: 'straight',
        style: { stroke: '#999', strokeWidth: 1 },
        markerEnd: ARROW_GRAY,
        data: { edgeKind: 'program_order' }
      };
    }

    const bulge = LANE_BASE + dep.lane * LANE_STEP;
    const isError = dep.kind === 'violation';

    return {
      id: baseId,
      source: dep.sourceId,
      target: dep.targetId,
      sourceHandle: 'right-source',
      targetHandle: 'right-target',
      type: 'bulge',
      data: { bulge, edgeKind: dep.kind },
      label: formatEdgeLabel(dep.locations),
      labelStyle: {
        fontSize: 10,
        fontFamily: 'monospace',
        fill: isError ? '#dc3545' : undefined
      },
      style: isError
        ? {
            stroke: '#dc3545',
            strokeWidth: 2,
            strokeDasharray: '5,5'
          }
        : { stroke: '#2472c8', strokeWidth: 2 },
      markerEnd: isError ? ARROW_RED : ARROW_BLUE
    };
  });

  return { nodes, allEdges };
}

// ============================================================================
// React component
// ============================================================================

interface IDependenciesDisplayProps {
  cellData: ICellGraphData[];
}

/**
 * Inner component with hover state for edge filtering.
 */
const FlowGraph: React.FC<{ cellData: ICellGraphData[] }> = ({ cellData }) => {
  const [hoveredNodeId, setHoveredNodeId] = React.useState<string | null>(null);

  const { nodes, allEdges } = React.useMemo(
    () => buildGraph(cellData),
    [cellData]
  );

  // Filter edges based on hover:
  //   - program_order: always visible
  //   - error edges (write_write, read_write): always visible
  //   - data_dependency: only visible when source or target is hovered
  const edges = React.useMemo(() => {
    return allEdges.filter(edge => {
      const kind = edge.data?.edgeKind;
      if (kind === 'program_order') {
        return true;
      }
      if (kind === 'violation') {
        return true;
      }
      // data_dependency: show only if hovered node is involved
      if (hoveredNodeId === null) {
        return false;
      }
      return edge.source === hoveredNodeId || edge.target === hoveredNodeId;
    });
  }, [allEdges, hoveredNodeId]);

  const onNodeMouseEnter: NodeMouseHandler = React.useCallback((_, node) => {
    setHoveredNodeId(node.id);
  }, []);

  const onNodeMouseLeave: NodeMouseHandler = React.useCallback(() => {
    setHoveredNodeId(null);
  }, []);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodeMouseEnter={onNodeMouseEnter}
      onNodeMouseLeave={onNodeMouseLeave}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      fitView={true}
      fitViewOptions={{ padding: 0.15, minZoom: 0.5, maxZoom: 1.2 }}
      minZoom={0.3}
      maxZoom={2}
    >
      <Controls showInteractive={false} />
    </ReactFlow>
  );
};

const DependenciesDisplay: React.FC<IDependenciesDisplayProps> = ({
  cellData
}) => {
  if (cellData.length === 0) {
    return (
      <div className="flowbook-metadata-empty">
        <p>No cells to display.</p>
        <p>Execute cells to see dependencies.</p>
      </div>
    );
  }

  return (
    <div
      className="flowbook-dependencies-graph"
      style={{ width: '100%', height: '100%', position: 'relative' }}
    >
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0
        }}
      >
        <ReactFlowProvider>
          <FlowGraph cellData={cellData} />
        </ReactFlowProvider>
      </div>
    </div>
  );
};

// ============================================================================
// Lumino Widget
// ============================================================================

export class DependenciesPanel extends Widget {
  private _cellData: ICellGraphData[] = [];

  constructor() {
    super();
    this.id = 'flowbook-dependencies-panel';
    this.addClass('flowbook-dependencies-panel');
    this.title.label = 'Dependencies';
    this.title.closable = true;
    this.title.caption = 'Cell dependency graph';
    this.render();
  }

  private render(): void {
    ReactDOM.render(
      <DependenciesDisplay cellData={this._cellData} />,
      this.node
    );
  }

  public updateGraph(cellData: ICellGraphData[]): void {
    this._cellData = cellData;
    this.render();
  }

  public clear(): void {
    this._cellData = [];
    this.render();
  }

  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    ReactDOM.unmountComponentAtNode(this.node);
    super.dispose();
  }
}
