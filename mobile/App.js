import React, { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

import * as ImagePicker from "expo-image-picker";
import * as DocumentPicker from "expo-document-picker";
import { Video, ResizeMode } from "expo-av";

const API_BASE_URL = "http://192.168.1.216:8000";

const getPhaseConfig = (exerciseLabel) => {
  const label = String(exerciseLabel || "").toLowerCase();

  if (label.includes("squat")) {
    return {
      text: "Setup → Bottom → Stand Tall",
        items: [
          ["setup", "Setup"],
          ["bottom", "Bottom ⭐"],
          ["stand", "Stand Tall"],
        ],
    };
  }

  if (label.includes("push press")) {
    return {
      text: "Dip → Drive → Lockout",
      highlight: "lockout",
      items: [
        ["dip", "Dip"],
        ["drive", "Drive"],
        ["lockout", "Lockout ⭐"],
      ],
    };
  }

  if (label.includes("bench")) {
    return {
      text: "Bottom → Press → Lockout",
      highlight: "lockout",
      items: [
        ["bottom", "Bottom"],
        ["press", "Press"],
        ["lockout", "Lockout ⭐"],
      ],
    };
  }

  return {
    text: "Setup → Pull → Lockout",
    highlight: "lockout",
    items: [
      ["setup", "Setup"],
      ["pull", "Pull"],
      ["lockout", "Lockout ⭐"],
    ],
  };
};

export default function App() {
  const [video, setVideo] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const reset = () => setResult(null);

  const recordWithCamera = async () => {
    reset();
    const permission = await ImagePicker.requestCameraPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("Camera permission required");
      return;
    }

    const res = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
      videoMaxDuration: 60,
    });

    if (!res.canceled) {
      const a = res.assets[0];
      setVideo({
        uri: a.uri,
        name: a.fileName || "camera.mov",
        type: a.mimeType || "video/quicktime",
      });
    }
  };

  const pickFromLibrary = async () => {
    reset();
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("Library permission required");
      return;
    }

    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
    });

    if (!res.canceled) {
      const a = res.assets[0];
      setVideo({
        uri: a.uri,
        name: a.fileName || "library.mov",
        type: a.mimeType || "video/quicktime",
      });
    }
  };

  const pickFromCloud = async () => {
    reset();

    const res = await DocumentPicker.getDocumentAsync({
      type: "video/*",
      copyToCacheDirectory: true,
    });

    if (!res.canceled) {
      const a = res.assets[0];
      setVideo({
        uri: a.uri,
        name: a.name || "cloud.mov",
        type: a.mimeType || "video/quicktime",
      });
    }
  };

  const analyzeVideo = async () => {
    if (!video) {
      Alert.alert("Pick a video first");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();

      formData.append("file", {
        uri: video.uri,
        name: video.name,
        type: video.type,
      });

      const res = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      console.log("BACKEND RESPONSE:", data);

      if (!res.ok) {
        throw new Error(data.detail || data.message || "Analyze request failed");
      }

      setResult(data);
    } catch (err) {
      setResult({ error: true, message: err.message });
    } finally {
      setLoading(false);
    }
  };

  const formatLabel = (v) =>
    v
      ? String(v)
          .replaceAll("_", " ")
          .replace(/\b\w/g, (c) => c.toUpperCase())
      : "N/A";

  const reps = result?.rep_feedback || [];

  const overallScore =
    reps.length > 0
      ? Math.round(
          reps.reduce((sum, r) => sum + Number(r.score || 0), 0) / reps.length
        )
      : null;

  const biggestFix =
    result?.set_summary?.biggest_fix ||
    result?.rep_feedback?.[0]?.feedback?.[0] ||
    "Keep building consistent reps.";

  const phaseConfig = getPhaseConfig(result?.exercise_label);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>FormCheck AI</Text>

        <View style={styles.buttons}>
          <TouchableOpacity style={styles.primary} onPress={recordWithCamera}>
            <Text style={styles.buttonText}>Record</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.secondary} onPress={pickFromLibrary}>
            <Text style={styles.buttonText}>Library</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.secondary} onPress={pickFromCloud}>
            <Text style={styles.buttonText}>Files</Text>
          </TouchableOpacity>
        </View>

        {video && (
          <View style={styles.previewCard}>
            <Text style={styles.previewLabel}>Selected Video</Text>
            <Text style={styles.previewName}>{video.name}</Text>

            <Video
              source={{ uri: video.uri }}
              style={styles.video}
              useNativeControls
              resizeMode={ResizeMode.CONTAIN}
            />
          </View>
        )}

        <TouchableOpacity
          style={[styles.analyze, loading && styles.disabled]}
          onPress={analyzeVideo}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Analyze</Text>
          )}
        </TouchableOpacity>

        {result?.error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorText}>{result.message}</Text>
          </View>
        )}

        {result?.feedback && result.rep_feedback?.length === 0 && !result.error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorTitle}>Video Not Usable</Text>
            <Text style={styles.errorText}>
              Camera angle is too close or unclear for reliable scoring.
            </Text>
            <Text style={styles.errorText}>How to fix:</Text>
            <Text style={styles.errorText}>• Move camera farther back</Text>
            <Text style={styles.errorText}>• Record from the side</Text>
            <Text style={styles.errorText}>
              • Keep bar, chest, elbows, and feet visible
            </Text>
          </View>
        )}

        {result && !result.error && (
          <View>
            <Text style={styles.exercise}>
              {result.analysis_mode === "poor_video_quality"
                ? "Video Not Usable"
                : result.exercise_label}
            </Text>

            {overallScore !== null && (
              <View style={styles.summaryCard}>
                <Text style={styles.summaryLabel}>Overall Score</Text>
                <Text style={styles.summaryScore}>{overallScore}/10</Text>
                <Text style={styles.biggestFix}>Biggest Fix: {biggestFix}</Text>
              </View>
            )}

            {result?.overlay_video_url && (
              <View style={styles.overlayCard}>
                <Text style={styles.overlayTitle}>Coached Replay</Text>

                <View style={styles.legendRow}>
                  <View style={styles.legendBadge}>
                    <View style={[styles.legendDot, { backgroundColor: "#22c55e" }]} />
                    <Text style={styles.legendText}>Your Movement</Text>
                  </View>

                  <View style={styles.legendBadge}>
                    <View style={[styles.legendDot, { backgroundColor: "#3b82f6" }]} />
                    <Text style={styles.legendText}>Ideal Form</Text>
                  </View>
                </View>

                <Text style={styles.overlayText}>What happened: {biggestFix}</Text>

                <Video
                  source={{ uri: `${API_BASE_URL}${result.overlay_video_url}` }}
                  style={styles.overlayVideo}
                  useNativeControls
                  shouldPlay={false}
                  resizeMode={ResizeMode.CONTAIN}
                />
              </View>
            )}

            {result?.phase_images && (
              <View style={styles.phaseCard}>
                <Text style={styles.phaseTitle}>Key Positions</Text>
                <Text style={styles.phaseText}>{phaseConfig.text}</Text>

                <View style={styles.legendRow}>
                  <View style={styles.legendBadge}>
                    <View style={[styles.legendDot, { backgroundColor: "#22c55e" }]} />
                    <Text style={styles.legendText}>Your Movement</Text>
                  </View>

                  <View style={styles.legendBadge}>
                    <View style={[styles.legendDot, { backgroundColor: "#3b82f6" }]} />
                    <Text style={styles.legendText}>Ideal Form</Text>
                  </View>
                </View>

                {phaseConfig.items.map(([key, label]) => {
                  const imageUrl = result.phase_images?.[key];
                  if (!imageUrl) return null;

                  return (
                    <View
                      key={key}
                      style={[
                        styles.phaseImageCard,
                        key === phaseConfig.highlight && styles.highlightCard,
                      ]}
                    >
                      <Text style={styles.phaseImageLabel}>{label}</Text>

                      <Image
                        source={{ uri: `${API_BASE_URL}${imageUrl}` }}
                        style={styles.phaseSingleImage}
                        resizeMode="contain"
                      />
                    </View>
                  );
                })}
              </View>
            )}

            {reps.map((rep, i) => (
              <View key={i} style={styles.card}>
                <Text style={styles.rep}>
                  Rep {rep.rep} — {rep.grade}
                </Text>

                <Text style={styles.score}>Score: {rep.score}/10</Text>

                <View style={styles.metrics}>
                  {Object.entries(rep.breakdown || {}).map(([key, value]) => (
                    <Text key={key} style={styles.metricText}>
                      {formatLabel(key)}: {formatLabel(value)}
                    </Text>
                  ))}
                </View>

                {rep.issues?.length > 0 && (
                  <>
                    <Text style={styles.section}>Issues</Text>
                    {rep.issues.map((x, j) => (
                      <Text key={j} style={styles.issue}>
                        • {x}
                      </Text>
                    ))}
                  </>
                )}

                {rep.feedback?.length > 0 && (
                  <>
                    <Text style={styles.section}>
                      {rep.issues?.length > 0 ? "What to Fix" : "Coach Note"}
                    </Text>

                    {rep.feedback.map((item, j) => (
                      <Text key={j} style={styles.coach}>
                        → {item}
                      </Text>
                    ))}
                  </>
                )}
              </View>
            ))}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0b1020" },
  content: { padding: 20, paddingBottom: 60 },

  title: { color: "#fff", fontSize: 32, fontWeight: "800" },
  buttons: { gap: 10, marginVertical: 20 },

  primary: {
    backgroundColor: "#2563eb",
    padding: 15,
    borderRadius: 12,
    alignItems: "center",
  },

  secondary: {
    backgroundColor: "#1f2937",
    padding: 15,
    borderRadius: 12,
    alignItems: "center",
  },

  analyze: {
    backgroundColor: "#16a34a",
    padding: 15,
    borderRadius: 12,
    alignItems: "center",
    marginBottom: 20,
  },

  disabled: { opacity: 0.6 },
  buttonText: { color: "#fff", fontWeight: "800" },

  previewCard: {
    backgroundColor: "#111827",
    padding: 12,
    borderRadius: 14,
    marginBottom: 15,
  },

  previewLabel: { color: "#9ca3af", fontSize: 12, marginBottom: 2 },
  previewName: { color: "#fff", fontWeight: "700", marginBottom: 10 },

  video: {
    width: "100%",
    height: 220,
    backgroundColor: "#000",
    borderRadius: 12,
    marginTop: 10,
  },

  overlayCard: {
    backgroundColor: "#052e16",
    padding: 16,
    borderRadius: 16,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: "#22c55e",
  },

  overlayTitle: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "900",
    marginBottom: 10,
  },

  overlayText: {
    color: "#86efac",
    fontSize: 16,
    fontWeight: "700",
    marginBottom: 12,
  },

  overlayVideo: {
    width: "100%",
    height: 300,
    backgroundColor: "#000",
    borderRadius: 12,
  },

  legendRow: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 12,
    flexWrap: "wrap",
  },

  legendBadge: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.08)",
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
  },

  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },

  legendText: { color: "#fff", fontSize: 14, fontWeight: "700" },

  phaseCard: {
    backgroundColor: "#111827",
    padding: 16,
    borderRadius: 16,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: "#3b82f6",
  },

  phaseTitle: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "900",
    marginBottom: 8,
  },

  phaseText: {
    color: "#bfdbfe",
    fontSize: 15,
    fontWeight: "700",
    marginBottom: 12,
  },

  phaseImageCard: {
    backgroundColor: "#020617",
    borderRadius: 14,
    padding: 10,
    marginTop: 14,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },

  highlightCard: {
    borderColor: "#fbbf24",
    borderWidth: 2,
  },

  phaseImageLabel: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "900",
    marginBottom: 8,
  },

  phaseSingleImage: {
    width: "100%",
    height: 280,
    backgroundColor: "#000",
    borderRadius: 12,
  },

  exercise: {
    color: "#fff",
    fontSize: 28,
    fontWeight: "900",
    marginBottom: 12,
  },

  summaryCard: {
    backgroundColor: "#111827",
    padding: 16,
    borderRadius: 16,
    marginBottom: 16,
  },

  summaryLabel: {
    color: "#9ca3af",
    fontSize: 13,
    fontWeight: "700",
    marginBottom: 4,
  },

  summaryScore: {
    color: "#fff",
    fontSize: 34,
    fontWeight: "900",
  },

  biggestFix: {
    color: "#86efac",
    fontSize: 16,
    fontWeight: "700",
    marginTop: 10,
  },

  card: {
    backgroundColor: "#1f2937",
    padding: 16,
    borderRadius: 16,
    marginBottom: 20,
  },

  rep: {
    color: "#fff",
    fontWeight: "900",
    fontSize: 20,
  },

  score: {
    color: "#fbbf24",
    fontSize: 18,
    fontWeight: "700",
    marginBottom: 12,
  },

  metrics: { marginBottom: 14 },

  metricText: {
    color: "#d1d5db",
    fontSize: 16,
    marginBottom: 3,
  },

  section: {
    color: "#fff",
    fontWeight: "900",
    fontSize: 17,
    marginTop: 6,
  },

  issue: {
    color: "#fca5a5",
    fontSize: 16,
    marginTop: 3,
  },

  coach: {
    color: "#86efac",
    fontSize: 16,
    marginTop: 5,
  },

  errorCard: {
    backgroundColor: "#7f1d1d",
    padding: 14,
    borderRadius: 12,
    marginBottom: 15,
  },

  errorText: { color: "#fff", marginTop: 4 },

  errorTitle: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "900",
    marginBottom: 10,
  },
});