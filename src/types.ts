/**
 * Type definitions for the Ferret JupyterLab extension
 */

/**
 * Information about a Ferret command from the server
 */
export interface CommandInfo {
  id: string;
  label: string;
  icon: string;
  tooltip: string;
  requires_kernel: boolean;
}

/**
 * History entry representing a single operation (command or user edit)
 */
export interface IHistoryEntry {
  id: string;
  timestamp: number;
  type: 'command' | 'user-edit';
  commandId?: string;
  commandLabel?: string;
  icon: string;
  notebookSnapshot: any;
  affectedCells: string[];
  addedCells?: string[];
  deletedCells?: string[];
  modifiedCells?: string[];
  metadata?: any;
  description: string;
  editSummary?: {
    cellsAdded: number;
    cellsDeleted: number;
    cellsModified: number;
    cellsMoved: number;
  };
}

/**
 * History state for a notebook
 */
export interface IHistoryState {
  entries: IHistoryEntry[];
  currentIndex: number;
  maxEntries: number;
  pendingEdit: boolean;
  lastSnapshotTime: number;
  editDebounceMs: number;
}

/**
 * Hardcoded list of available Ferret commands
 */
export const FERRET_COMMANDS: CommandInfo[] = [
  {
    id: 'cleanup',
    label: 'Tidy',
    icon: 'ui-components:code',
    tooltip: 'Generate improved code with AI suggestions',
    requires_kernel: false
  },
  {
    id: 'inspect',
    label: 'Inspect',
    icon: 'ui-components:search',
    tooltip: 'Inspect for optimization and potential issues',
    requires_kernel: false
  },
  {
    id: 'optimize',
    label: 'Optimize',
    icon: 'ui-components:flash',
    tooltip: 'Optimize cells based on inspection metadata',
    requires_kernel: true
  },
  {
    id: 'document',
    label: 'Document',
    icon: 'ui-components:text-editor',
    tooltip: 'Add documentation comments to code cells',
    requires_kernel: false
  },
  {
    id: 'profile',
    label: 'Profile',
    icon: 'ui-components:info',
    tooltip: 'Profile code cells',
    requires_kernel: true
  },
  {
    id: 'validate_change',
    label: 'Validate Change',
    icon: 'ui-components:check',
    tooltip: 'Validate selected cells with next cell comparison',
    requires_kernel: true
  },
  {
    id: 'generate',
    label: 'Generate',
    icon: 'ui-components:code',
    tooltip: 'Generate code from string specification',
    requires_kernel: false
  },
  {
    id: 'test',
    label: 'Test',
    icon: 'ui-components:check',
    tooltip: 'Run unit tests for cell(s)',
    requires_kernel: true
  },
  {
    id: 'generate_tests',
    label: 'Generate Tests',
    icon: 'ui-components:build',
    tooltip: 'Auto-generate unit tests for cell(s)',
    requires_kernel: false
  }
];

/**
 * Result from executing a command
 */
export interface CommandResult {
  notebook: any;
  metadata: any;
  total_cost: number;
  total_time: number;
}

/**
 * Information about a Jupyter kernel
 */
export interface KernelInfo {
  kernel_id: string;
  kernel_name: string;
  connection_file?: string;
}

/**
 * Request body for the /ferret/execute API endpoint
 */
export interface ExecuteCommandRequest {
  command: string;
  notebook: any;
  kernel_id?: string;
  params?: Record<string, any>;
  selected_cell_ids?: string[];
}

/**
 * Ferret optimization step structure
 */
export interface IFerretOptimizationStep {
  target_cell_id: string;
  function_name: string | null;
  description: string[];
}

/**
 * Ferret optimization potential metadata structure
 */
export interface IFerretOptimizationPotential {
  potential: number;
  optimization_plan: IFerretOptimizationStep[];
}

/**
 * Ferret profile metadata structure
 */
export interface IFerretProfileData {
  duration: number;
  profile: string;
  env: Record<string, string>;
  env_after: Record<string, string>;
}

/**
 * Ferret generated code metadata structure
 */
export interface IFerretGeneratedCodeMetadata {
  explanation: string;
  original_spec: string;
}

/**
 * Ferret optimized code metadata structure
 */
export interface IFerretOptimizedCodeMetadata {
  original_code: string;
  optimized_code: string;
  optimizations_applied: string[];
}

/**
 * Ferret optimization applied metadata structure
 */
export interface IFerretOptimizationAppliedMetadata {
  modified_cell_ids: string[];
}

/**
 * Unit test structure for a cell
 */
export interface IUnitTest {
  title: string;
  description: string;
  setup_code: string;
  assertion_code: string;
}

/**
 * Unit tests collection for a cell
 */
export interface IUnitTests {
  tests: IUnitTest[];
}

/**
 * Ferret cell metadata structure
 */
export interface IFerretMetadata {
  optimization_potential?: IFerretOptimizationPotential;
  profile?: IFerretProfileData;
  generated?: IFerretGeneratedCodeMetadata;
  optimized?: IFerretOptimizedCodeMetadata;
  optimization_applied?: IFerretOptimizationAppliedMetadata;
  unit_tests?: IUnitTests;
}
