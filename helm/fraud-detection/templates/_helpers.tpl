{{/*
Expand the chart name.
*/}}
{{- define "fraud-detection.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Full image reference for a given service.
Usage: include "fraud-detection.image" (dict "root" . "repo" .Values.mlflow.image.repository "pullPolicy" .Values.mlflow.image.pullPolicy)
*/}}
{{- define "fraud-detection.image" -}}
{{ .root.Values.global.dockerhubUsername }}/{{ .repo }}:{{ .root.Values.global.imageTag }}
{{- end }}

{{/*
Common labels applied to every resource.
*/}}
{{- define "fraud-detection.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Selector labels for a named component.
Usage: include "fraud-detection.selectorLabels" (dict "component" "flask")
*/}}
{{- define "fraud-detection.selectorLabels" -}}
app: {{ .component }}
{{- end }}
